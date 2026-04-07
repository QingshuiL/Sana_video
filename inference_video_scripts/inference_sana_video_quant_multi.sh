#!/bin/bash
set -e

NP="${NP:-8}"

bash inference_video_scripts/inference_sana_video.sh \
  --np "$NP" \
  --fake_quant.enable_fake_quant=True \
  --fake_quant.enable_weight_fake_quant=True \
  --fake_quant.enable_activation_fake_quant=True \
  --fake_quant.linear.quant_format=mxfp \
  --fake_quant.linear.bit_width=4 \
  --fake_quant.linear.mxfp_impl=floor \
  --fake_quant.linear.granularity=per_token \
  --fake_quant.conv.quant_format=mxfp \
  --fake_quant.conv.bit_width=8 \
  --fake_quant.conv.mxfp_impl=floor \
  "$@"
