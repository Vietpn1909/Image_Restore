#!/bin/bash
# Download pretrained weights for Real-ESRGAN and GFPGAN
set -e

mkdir -p checkpoints/pretrained
cd checkpoints/pretrained

echo "Downloading Real-ESRGAN x4plus..."
wget -nc -q --show-progress \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

echo "Downloading GFPGAN v1.4..."
wget -nc -q --show-progress \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

echo "Done."
ls -lh
