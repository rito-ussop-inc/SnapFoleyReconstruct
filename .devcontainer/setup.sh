#!/usr/bin/env bash
# One-time Codespace setup: system tools + python deps (runs automatically on container creation)
set -e
echo "[SnapFoley] installing ffmpeg..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg >/dev/null
echo "[SnapFoley] installing CPU torch (small + fast)..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "[SnapFoley] installing python deps..."
pip install -q -r requirements-cloud.txt || pip install -q -r requirements-colab.txt
chmod +x start_snapfoley.sh 2>/dev/null || true
echo "[SnapFoley] setup complete - run ./start_snapfoley.sh to launch"
