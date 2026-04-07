## Inference CLI

### Inference SANA-Video

```bash
python app/sana_video_pipeline.py \
        --config configs/sana_video_config/480ms/Sana_1600M_480px_adamW_fsdp.yaml \
        --model_path "hf://Efficient-Large-Model/SanaVideo_willquant/checkpoints/model.pth" \
        --save_path sana_video.mp4 \
        --prompt "In a whimsical forest setting, a small deer with antlers stands amidst oversized mushrooms and scattered carrots. The scene is vibrant with lush green moss and rocks, creating a magical atmosphere. The deer appears curious, moving slowly across the ground, surrounded by the towering fungi and colorful vegetables. The sky above is clear and bright, adding to the enchanting ambiance. A low-angle shot captures the deer's gentle exploration of this fantastical landscape."
```

### Inference SANA-Video Chunked Version

```bash
python app/sana_video_pipeline.py \
        --config configs/sana_video_config/480ms/Sana_1600M_480px_adamW_fsdp_chunk.yaml \
        --model_path "hf://Efficient-Large-Model/SanaVideo_chunk/checkpoints/model.pth" \
        --save_path sana_video_chunk_i2v.mp4 \
        --interval_k 0.2 \
        --image_path output/tmp_videos/wan_goodcase_i2v_eval/00000000_video_001.jpg \
        --prompt "In a whimsical forest setting, a small deer with antlers stands amidst oversized mushrooms and scattered carrots. The scene is vibrant with lush green moss and rocks, creating a magical atmosphere. The deer appears curious, moving slowly across the ground, surrounded by the towering fungi and colorful vegetables. The sky above is clear and bright, adding to the enchanting ambiance. A low-angle shot captures the deer's gentle exploration of this fantastical landscape."
```

### MXFP Fake Quantization

The video inference path supports block-only fake quantization for `Linear` and `Conv2d` layers.

- `mxfp` now has two implementations:
- `variant`: local implementation that rounds the shared exponent up when the block mantissa is large
- `floor`: implementation aligned with the official `torchao` MX `FLOOR` scaling rule

Relevant inference config fields:

- `fake_quant.enable_fake_quant`
- `fake_quant.enable_weight_fake_quant`
- `fake_quant.enable_activation_fake_quant`
- `fake_quant.linear.quant_format`
- `fake_quant.linear.bit_width`
- `fake_quant.linear.mxfp_impl`
- `fake_quant.linear.granularity`
- `fake_quant.conv.quant_format`
- `fake_quant.conv.bit_width`
- `fake_quant.conv.mxfp_impl`

Example:

```yaml
fake_quant:
  enable_fake_quant: true
  enable_weight_fake_quant: true
  enable_activation_fake_quant: true
  linear:
    quant_format: mxfp
    bit_width: 4
    mxfp_impl: floor
    granularity: per_token
  conv:
    quant_format: mxfp
    bit_width: 8
    mxfp_impl: floor
```

Quantized inference helper scripts:

- `inference_video_scripts/inference_sana_video_quant_single.sh`
- `inference_video_scripts/inference_sana_video_quant_multi.sh`

Single GPU example:

```bash
bash inference_video_scripts/inference_sana_video_quant_single.sh \
  --config configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
  --cfg_scale 6 \
  --motion_score 30 \
  --flow_shift 8 \
  --work_dir output/sana_video_quant_single
```

Multi GPU example:

```bash
NP=8 bash inference_video_scripts/inference_sana_video_quant_multi.sh \
  --config configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
  --cfg_scale 6 \
  --motion_score 30 \
  --flow_shift 8 \
  --work_dir output/sana_video_quant_multi
```
