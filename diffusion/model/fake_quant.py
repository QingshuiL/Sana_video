from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


QuantFormat = Literal["int", "mxfp", "nvfp"]
LinearGranularity = Literal["per_token", "per_channel"]
FP32_MIN_NORMAL = 2 ** (-126)
MXFP_SCALE_BITS = 8
MXFP_BLOCK_SIZE = 32


@dataclass
class FakeQuantConfig:
    enable_fake_quant: bool = False
    linear_quant_format: QuantFormat = "int"
    conv_quant_format: QuantFormat = "int"
    linear_bit_width: int = 4
    conv_bit_width: int = 8
    enable_weight_fake_quant: bool = True
    enable_activation_fake_quant: bool = True
    linear_granularity: LinearGranularity = "per_token"
    conv_activation_granularity: str = "per_tensor"
    conv_weight_granularity: str = "per_channel"

    def validate(self) -> None:
        valid_formats = {"int", "mxfp", "nvfp"}
        if self.linear_quant_format not in valid_formats:
            raise ValueError(f"Unsupported linear quant format: {self.linear_quant_format}")
        if self.conv_quant_format not in valid_formats:
            raise ValueError(f"Unsupported conv quant format: {self.conv_quant_format}")
        if self.linear_granularity not in {"per_token", "per_channel"}:
            raise ValueError(f"Unsupported linear granularity: {self.linear_granularity}")
        if self.conv_activation_granularity != "per_tensor":
            raise ValueError("Conv2d activation granularity must be per_tensor")
        if self.conv_weight_granularity != "per_channel":
            raise ValueError("Conv2d weight granularity must be per_channel")
        if self.linear_bit_width < 2:
            raise ValueError("linear_bit_width must be >= 2")
        if self.conv_bit_width < 2:
            raise ValueError("conv_bit_width must be >= 2")
        if self.linear_quant_format == "nvfp" and self.linear_bit_width < 4:
            raise ValueError("nvfp linear quantization requires linear_bit_width >= 4")
        if self.conv_quant_format == "nvfp" and self.conv_bit_width < 4:
            raise ValueError("nvfp conv quantization requires conv_bit_width >= 4")
        if self.linear_quant_format == "mxfp" and self.linear_bit_width not in {4, 8}:
            raise ValueError("mxfp linear quantization only supports bit_width=4 (fp4_e2m1) or bit_width=8 (fp8_e4m3)")
        if self.conv_quant_format == "mxfp" and self.conv_bit_width not in {4, 8}:
            raise ValueError("mxfp conv quantization only supports bit_width=4 (fp4_e2m1) or bit_width=8 (fp8_e4m3)")


def _safe_absmax(x: torch.Tensor, dims: Sequence[int] | None = None) -> torch.Tensor:
    if dims is None:
        return x.detach().abs().amax().clamp_min(1e-12)
    return x.detach().abs().amax(dim=tuple(dims), keepdim=True).clamp_min(1e-12)


def _round_nearest_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _safe_lshift(x: torch.Tensor, bits: int, exp: torch.Tensor | None) -> torch.Tensor:
    if exp is None:
        return x * (2**bits)
    return x / torch.pow(torch.full_like(x, 2.0), exp) * (2**bits)


def _safe_rshift(x: torch.Tensor, bits: int, exp: torch.Tensor | None) -> torch.Tensor:
    if exp is None:
        return x / (2**bits)
    return x / (2**bits) * torch.pow(torch.full_like(x, 2.0), exp)


def _mxfp_elem_format_params(bit_width: int) -> tuple[int, int, int, float]:
    if bit_width == 4:
        return 2, 3, 2, 2**2 * 1.5
    if bit_width == 8:
        return 4, 5, 8, 2**8 * 1.75
    raise ValueError("mxfp only supports bit_width=4 (fp4_e2m1) or bit_width=8 (fp8_e4m3)")


def _shared_exponents(x: torch.Tensor, dims: Sequence[int] | None = None) -> torch.Tensor:
    if dims is None:
        shared = torch.max(torch.abs(x))
    else:
        shared = torch.abs(x).amax(dim=tuple(dims), keepdim=True)
    zero_mask = (shared == 0).to(shared.dtype)
    return torch.floor(torch.log2(shared + FP32_MIN_NORMAL * zero_mask))


def _quantize_elemwise_core_mxfp(
    x: torch.Tensor,
    bits: int,
    exp_bits: int,
    max_norm: float,
) -> torch.Tensor:
    private_exp = torch.floor(torch.log2(torch.abs(x) + (x == 0).to(x.dtype)))
    min_exp = -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clamp(min=min_exp)

    quantized = _safe_lshift(x, bits - 2, private_exp)
    quantized = _round_nearest_away_from_zero(quantized)
    quantized = _safe_rshift(quantized, bits - 2, private_exp)
    quantized = torch.clamp(quantized, min=-max_norm, max=max_norm)

    quantized = torch.where(x == float("Inf"), torch.full_like(quantized, float("Inf")), quantized)
    quantized = torch.where(x == -float("Inf"), torch.full_like(quantized, -float("Inf")), quantized)
    quantized = torch.where(torch.isnan(x), torch.full_like(quantized, float("NaN")), quantized)
    return quantized


def _reshape_to_blocks(
    x: torch.Tensor,
    dims: Sequence[int],
    block_size: int,
) -> tuple[torch.Tensor, list[int], torch.Size, torch.Size]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    axes = sorted(d + x.ndim if d < 0 else d for d in dims)
    reshaped = x
    for idx, axis in enumerate(axes):
        axes[idx] = axis + idx
        reshaped = torch.unsqueeze(reshaped, dim=axes[idx] + 1)

    orig_shape = reshaped.shape
    pad = [0] * (2 * len(orig_shape))
    do_padding = False
    for axis in axes:
        axis_len = orig_shape[axis]
        if axis_len % block_size == 0:
            continue
        pad[2 * axis] = block_size - axis_len % block_size
        do_padding = True

    if do_padding:
        reshaped = F.pad(reshaped, list(reversed(pad)), mode="constant")

    padded_shape = reshaped.shape
    target_shape = list(padded_shape)
    for axis in axes:
        if target_shape[axis] >= block_size:
            target_shape[axis + 1] = block_size
            target_shape[axis] = target_shape[axis] // block_size
        else:
            target_shape[axis + 1] = target_shape[axis]
            target_shape[axis] = 1

    reshaped = reshaped.reshape(target_shape)
    return reshaped, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(
    x: torch.Tensor,
    padded_shape: torch.Size,
    orig_shape: torch.Size,
    axes: Sequence[int],
) -> torch.Tensor:
    restored = x.reshape(padded_shape)
    index = [slice(None)] * restored.ndim
    for axis in axes:
        index[axis] = slice(0, orig_shape[axis])
    restored = restored[tuple(index)]
    squeeze_dims = [axis + 1 for axis in axes]
    for axis in reversed(squeeze_dims):
        restored = restored.squeeze(axis)
    return restored


def _block_mxfp_quant(
    x: torch.Tensor,
    bit_width: int,
    dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    MXFP quantization aligned with microxcaling for fp4_e2m1 and fp8_e4m3.

    The tensor shares one exponent across the reduction axes, then each element
    is quantized with the selected floating-point element format before being
    rescaled back to fp32.
    """

    exp_bits, mantissa_bits_with_sign, emax, max_norm = _mxfp_elem_format_params(bit_width)
    quantized_input = x
    shared_exp_dims = dims
    block_axes: list[int] | None = None
    padded_shape: torch.Size | None = None
    orig_shape: torch.Size | None = None

    if dims is not None:
        quantized_input, block_axes, orig_shape, padded_shape = _reshape_to_blocks(x, dims=dims, block_size=MXFP_BLOCK_SIZE)
        shared_exp_dims = [axis + 1 for axis in block_axes]

    shared_exp = _shared_exponents(quantized_input, dims=shared_exp_dims) - emax

    scale_emax = 2 ** (MXFP_SCALE_BITS - 1) - 1
    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float("NaN")), shared_exp)
    shared_exp = torch.clamp(shared_exp, min=-scale_emax)

    scale = torch.pow(torch.full_like(shared_exp, 2.0), shared_exp)
    quantized = _quantize_elemwise_core_mxfp(
        quantized_input / scale,
        bits=mantissa_bits_with_sign,
        exp_bits=exp_bits,
        max_norm=max_norm,
    )
    quantized = quantized * scale

    if dims is not None and block_axes is not None and padded_shape is not None and orig_shape is not None:
        quantized = _undo_reshape_to_blocks(quantized, padded_shape=padded_shape, orig_shape=orig_shape, axes=block_axes)
    return quantized


def _resolve_nvfp_layout(bit_width: int) -> tuple[int, int]:
    """
    Map a total bit width to a floating-point layout.

    We keep 1 sign bit and split the remainder across exponent/mantissa. The
    mapping favors a small mantissa for low bit widths and preserves a valid
    exponent field for all supported widths.
    """

    if bit_width < 4:
        raise ValueError("nvfp quantization requires bit_width >= 4")
    exp_bits = max(2, bit_width // 2)
    exp_bits = min(exp_bits, bit_width - 2)
    mantissa_bits = bit_width - 1 - exp_bits
    if mantissa_bits < 1:
        raise ValueError(f"nvfp bit_width={bit_width} does not leave room for mantissa bits")
    return exp_bits, mantissa_bits


def _quantize_finite_float(
    x: torch.Tensor,
    exp_bits: int,
    mantissa_bits: int,
) -> torch.Tensor:
    """
    Quantize a tensor to a restricted finite floating-point format around 1.0.

    The caller is responsible for any outer block scaling. This function handles
    normalized numbers, exponent saturation, and subnormals in an elementwise way.
    """

    x_abs = x.abs()
    x_sign = torch.sign(x)

    exponent_bias = 2 ** (exp_bits - 1) - 1
    exponent_min = 1 - exponent_bias
    exponent_max = (2**exp_bits - 2) - exponent_bias
    min_normal = 2.0 ** exponent_min
    max_finite = (2.0 - 2.0 ** (-mantissa_bits)) * (2.0 ** exponent_max)

    x_abs = x_abs.clamp(max=max_finite)
    is_normal = x_abs >= min_normal

    normal_exponent = torch.floor(torch.log2(x_abs.clamp_min(min_normal)))
    normal_exponent = normal_exponent.clamp(min=exponent_min, max=exponent_max)
    normal_mantissa = x_abs / (2.0 ** normal_exponent) - 1.0
    normal_mantissa = normal_mantissa.clamp(0.0, 1.0 - 2.0 ** (-mantissa_bits))
    normal_mantissa = _round_nearest_away_from_zero(normal_mantissa * (2.0**mantissa_bits)) / (2.0**mantissa_bits)
    normal_value = (1.0 + normal_mantissa) * (2.0 ** normal_exponent)

    subnormal_value = x_abs / min_normal
    subnormal_value = subnormal_value.clamp(0.0, 1.0 - 2.0 ** (-mantissa_bits))
    subnormal_value = _round_nearest_away_from_zero(subnormal_value * (2.0**mantissa_bits)) / (2.0**mantissa_bits)
    subnormal_value = subnormal_value * min_normal

    quantized = torch.where(is_normal, normal_value, subnormal_value)
    return x_sign * quantized

def quantize_mx_fix( A, quant_bit=4, scale_bits=8, block_size=32, ): 
    """Function used for MX* quantization""" 
    assert scale_bits > 0 
    if quant_bit == 8:
        ebits, mbits, emax, max_norm = 4, 5, 8, 448.0 
    # e4m3 min_exp = -6 elif quant_bit == 4: ebits, mbits, emax, max_norm = 2, 3, 2, 6.0 # e2m1 min_exp = 0 else: raise Exception("quant_bit must be 8 or 4") # mantissa fractional quant scale # e4m3 -> 2^3 = 8 # e2m1 -> 2^1 = 2 mantissa_scale = 2 ** (mbits - 2) # shared scale exponent range scale_emax = 2 ** (scale_bits - 1) - 1 ori_shape = A.shape last_dim = ori_shape[-1] # -------- pad to block_size on last dim -------- pad_size = (-last_dim) % block_size if pad_size != 0: A = F.pad(A, (0, pad_size)) padded_shape = A.shape # flatten all leading dims, keep block on last dim A = A.reshape(-1, block_size) # ------------------------- # 1) shared exponent # ------------------------- abs_A = torch.abs(A) shared_abs_max = torch.amax(abs_A, dim=-1, keepdim=True) # safe log2 for zero blocks safe_shared = torch.where( shared_abs_max > 0, shared_abs_max, torch.ones_like(shared_abs_max) ) # round2decimal - 保留你的进位优化 exponent = torch.floor(torch.log2(safe_shared)) mantissa = safe_shared / torch.exp2(exponent) shared_exp = torch.where(mantissa > 1.75, exponent + 1, exponent) # all-zero block -> exponent set to 0 shared_exp = torch.where(shared_abs_max > 0, shared_exp, torch.zeros_like(shared_exp)) # shift by element-format emax shared_exp = shared_exp - emax # clamp to scale_bits range shared_exp = shared_exp.clamp(min=-scale_emax, max=scale_emax) # apply shared scaling A = A / torch.exp2(shared_exp) # ------------------------- # 2) private exponent # ------------------------- abs_A = torch.abs(A) safe_abs_A = torch.where(abs_A > 0, abs_A, torch.ones_like(abs_A)) private_exp = torch.floor(torch.log2(safe_abs_A)) private_exp = private_exp.clamp(min=min_exp) private_exp = torch.exp2(private_exp) # ------------------------- # 3) mantissa quantization # ------------------------- A = A / private_exp * mantissa_scale A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5) # round-half-up A = A / mantissa_scale * private_exp # saturate to representable finite range A = torch.clamp(A, min=-max_norm, max=max_norm) # undo shared scaling A = A * torch.exp2(shared_exp) # -------- restore shape and remove pad -------- A = A.reshape(padded_shape) if pad_size != 0: A = A[..., :last_dim] return A


def _nvfp_quant(
    x: torch.Tensor,
    bit_width: int,
    dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    NVFP-style emulation with explicit exponent/mantissa rounding.

    This is intentionally different from MXFP:
    - MXFP uses a shared exponent per block and a fixed-point mantissa.
    - NVFP uses a shared block scale plus per-element exponent/mantissa rounding
      with saturation and subnormal handling.
    """

    exp_bits, mantissa_bits = _resolve_nvfp_layout(bit_width)
    max_finite = (2.0 - 2.0 ** (-mantissa_bits)) * (2.0 ** ((2**exp_bits - 2) - (2 ** (exp_bits - 1) - 1)))
    absmax = _safe_absmax(x, dims)
    # NVFP uses a shared scale for a block/group, but elements inside the block
    # still get their own limited exponent+mantissa representation.
    block_scale = (absmax / max_finite).clamp_min(1e-12)
    normalized = x / block_scale
    quantized = _quantize_finite_float(normalized, exp_bits=exp_bits, mantissa_bits=mantissa_bits)
    return quantized * block_scale


def fake_quant_int(x: torch.Tensor, bit_width: int, dims: Sequence[int] | None = None) -> torch.Tensor:
    qmax = float(2 ** (bit_width - 1) - 1)
    scale = _safe_absmax(x, dims) / qmax
    q = _round_nearest_away_from_zero((x / scale).clamp(-qmax - 1, qmax))
    return q * scale


def fake_quant_mxfp(x: torch.Tensor, bit_width: int, dims: Sequence[int] | None = None) -> torch.Tensor:
    return _block_mxfp_quant(x, bit_width=bit_width, dims=dims)


def fake_quant_nvfp(x: torch.Tensor, bit_width: int, dims: Sequence[int] | None = None) -> torch.Tensor:
    return _nvfp_quant(x, bit_width=bit_width, dims=dims)


def fake_quantize_tensor(
    x: torch.Tensor,
    quant_format: QuantFormat,
    bit_width: int,
    dims: Sequence[int] | None = None,
) -> torch.Tensor:
    if quant_format == "int":
        return fake_quant_int(x, bit_width=bit_width, dims=dims)
    if quant_format == "mxfp":
        return fake_quant_mxfp(x, bit_width=bit_width, dims=dims)
    if quant_format == "nvfp":
        return fake_quant_nvfp(x, bit_width=bit_width, dims=dims)
    raise ValueError(f"Unsupported quant format: {quant_format}")


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, config: FakeQuantConfig):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=linear.weight.requires_grad)
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)
            if linear.bias is not None
            else None
        )
        self.config = copy.deepcopy(config)

    @classmethod
    def from_float(cls, linear: nn.Linear, config: FakeQuantConfig) -> "FakeQuantLinear":
        return cls(linear, config)

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.enable_activation_fake_quant:
            return x
        if self.config.linear_granularity == "per_token":
            dims = [x.ndim - 1]
        else:
            dims = list(range(x.ndim - 1))
        return fake_quantize_tensor(
            x,
            quant_format=self.config.linear_quant_format,
            bit_width=self.config.linear_bit_width,
            dims=dims,
        )

    def _quantize_weight(self) -> torch.Tensor:
        if not self.config.enable_weight_fake_quant:
            return self.weight
        return fake_quantize_tensor(
            self.weight,
            quant_format=self.config.linear_quant_format,
            bit_width=self.config.linear_bit_width,
            dims=[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._quantize_activation(x)
        weight = self._quantize_weight()
        return F.linear(x, weight, self.bias)


class FakeQuantConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, config: FakeQuantConfig):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=conv.weight.requires_grad)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone(), requires_grad=conv.bias.requires_grad)
            if conv.bias is not None
            else None
        )
        self.config = copy.deepcopy(config)

    @classmethod
    def from_float(cls, conv: nn.Conv2d, config: FakeQuantConfig) -> "FakeQuantConv2d":
        return cls(conv, config)

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.enable_activation_fake_quant:
            return x
        return fake_quantize_tensor(
            x,
            quant_format=self.config.conv_quant_format,
            bit_width=self.config.conv_bit_width,
            dims=None,
        )

    def _quantize_weight(self) -> torch.Tensor:
        if not self.config.enable_weight_fake_quant:
            return self.weight
        return fake_quantize_tensor(
            self.weight,
            quant_format=self.config.conv_quant_format,
            bit_width=self.config.conv_bit_width,
            dims=[1, 2, 3],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._quantize_activation(x)
        weight = self._quantize_weight()
        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def _replace_modules_in_block(
    module: nn.Module,
    config: FakeQuantConfig,
    block_idx: int,
    prefix: str,
    replacements: List[Dict[str, str]],
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            _set_child_module(module, child_name, FakeQuantLinear.from_float(child, config))
            replacements.append(
                {
                    "block": str(block_idx),
                    "module": full_name,
                    "type": "Linear",
                    "format": config.linear_quant_format,
                    "bits": str(config.linear_bit_width),
                    "granularity": config.linear_granularity,
                }
            )
            continue
        if isinstance(child, nn.Conv2d):
            _set_child_module(module, child_name, FakeQuantConv2d.from_float(child, config))
            replacements.append(
                {
                    "block": str(block_idx),
                    "module": full_name,
                    "type": "Conv2d",
                    "format": config.conv_quant_format,
                    "bits": str(config.conv_bit_width),
                    "granularity": "act:per_tensor,weight:per_channel",
                }
            )
            continue
        _replace_modules_in_block(child, config, block_idx, full_name, replacements)


def apply_fake_quant_to_sana_video_blocks(model: nn.Module, config: FakeQuantConfig) -> List[Dict[str, str]]:
    config.validate()
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not expose a blocks attribute")
    replacements: List[Dict[str, str]] = []
    for block_idx, block in enumerate(model.blocks):
        for scope_name in ("attn", "cross_attn", "mlp"):
            scope = getattr(block, scope_name, None)
            if scope is None:
                continue
            _replace_modules_in_block(scope, config, block_idx, scope_name, replacements)
    return replacements
