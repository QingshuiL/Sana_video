from __future__ import annotations

import torch

from diffusion.model.fake_quant import MXFP_SCALE_BITS, fake_quant_mxfp


FP32_MIN_NORMAL = 2 ** (-126)


def _reference_round_nearest_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _reference_safe_lshift(x: torch.Tensor, bits: int, exp: torch.Tensor | None) -> torch.Tensor:
    if exp is None:
        return x * (2**bits)
    return x / torch.pow(torch.full_like(x, 2.0), exp) * (2**bits)


def _reference_safe_rshift(x: torch.Tensor, bits: int, exp: torch.Tensor | None) -> torch.Tensor:
    if exp is None:
        return x / (2**bits)
    return x / (2**bits) * torch.pow(torch.full_like(x, 2.0), exp)


def _reference_format(bit_width: int) -> tuple[int, int, int, float]:
    if bit_width == 4:
        return 2, 3, 2, 2**2 * 1.5
    if bit_width == 8:
        return 4, 5, 8, 2**8 * 1.75
    raise ValueError(bit_width)


def _reference_mxfp(x: torch.Tensor, bit_width: int, dims: list[int] | None) -> torch.Tensor:
    exp_bits, bits, emax, max_norm = _reference_format(bit_width)
    if dims is None:
        shared = torch.max(torch.abs(x))
    else:
        shared = torch.abs(x).amax(dim=tuple(dims), keepdim=True)
    shared_exp = torch.floor(torch.log2(shared + FP32_MIN_NORMAL * (shared == 0).to(shared.dtype))) - emax

    scale_emax = 2 ** (MXFP_SCALE_BITS - 1) - 1
    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float("NaN")), shared_exp)
    shared_exp = torch.clamp(shared_exp, min=-scale_emax)
    scale = torch.pow(torch.full_like(shared_exp, 2.0), shared_exp)

    normalized = x / scale
    private_exp = torch.floor(torch.log2(torch.abs(normalized) + (normalized == 0).to(normalized.dtype)))
    min_exp = -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clamp(min=min_exp)

    quantized = _reference_safe_lshift(normalized, bits - 2, private_exp)
    quantized = _reference_round_nearest_away_from_zero(quantized)
    quantized = _reference_safe_rshift(quantized, bits - 2, private_exp)
    quantized = torch.clamp(quantized, min=-max_norm, max=max_norm)

    quantized = torch.where(normalized == float("Inf"), torch.full_like(quantized, float("Inf")), quantized)
    quantized = torch.where(normalized == -float("Inf"), torch.full_like(quantized, -float("Inf")), quantized)
    quantized = torch.where(torch.isnan(normalized), torch.full_like(quantized, float("NaN")), quantized)
    return quantized * scale


def test_fake_quant_mxfp_fp4_matches_reference() -> None:
    x = torch.tensor(
        [[0.1, 0.3, 0.5, 0.9], [1.1, 1.6, 2.5, 3.5]],
        dtype=torch.float32,
    )
    expected = _reference_mxfp(x, bit_width=4, dims=[1])
    actual = fake_quant_mxfp(x, bit_width=4, dims=[1])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fake_quant_mxfp_fp8_matches_reference() -> None:
    x = torch.tensor(
        [[-300.0, -17.25, -0.9, 0.0], [0.125, 3.25, 18.5, 129.0]],
        dtype=torch.float32,
    )
    expected = _reference_mxfp(x, bit_width=8, dims=[1])
    actual = fake_quant_mxfp(x, bit_width=8, dims=[1])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fake_quant_mxfp_rounding_is_not_bankers_rounding() -> None:
    x = torch.tensor([1.125, -1.125], dtype=torch.float32)
    actual = fake_quant_mxfp(x, bit_width=8, dims=[0])
    expected = _reference_mxfp(x, bit_width=8, dims=[0])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
