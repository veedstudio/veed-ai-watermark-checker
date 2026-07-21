#!/usr/bin/env bash
# Install Meta VideoSeal itself for the veed-videoseal package.
#
# VideoSeal declares `decord`, which has no macOS arm64 wheel (and unreliable py3.12 wheels)
# and is unused for embed/detect on tensors — we do video I/O via ffmpeg. So we install
# VideoSeal with `--no-deps`. Its real runtime deps are NOT listed here: they live in
# pyproject.toml's `videoseal` extra; install them with `pip install ".[videoseal]"` (see
# README). This script is only the `--no-deps` peer-install logic.
set -euo pipefail

PIP="${PIP:-pip}"

# torch/torchvision are a hard dependency of this package, but some environments pin a CUDA
# build separately. Set SKIP_TORCH=1 to leave the existing torch untouched.
if [ "${SKIP_TORCH:-0}" != "1" ]; then
  $PIP install torch torchvision
fi

$PIP install --no-deps videoseal

echo "VideoSeal installed (without decord). Its runtime deps come from 'pip install .[videoseal]'."
echo "Weights download on first use, or pre-place y_256b_img.pth and set VEED_VIDEOSEAL_CKPT_DIR."
