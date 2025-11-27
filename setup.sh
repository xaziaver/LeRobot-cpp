#!/bin/bash
set -e

echo "=== LeRobot C++ – Setup ==="

# 1. Remove old LibTorch if exists
if [ -d "$HOME/libtorch" ]; then
    echo "Removing old ~/libtorch..."
    rm -rf "$HOME/libtorch"
fi

# 2. Download latest shared LibTorch (CPU)
VERSION="2.5.0"
URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcpu.zip"
echo "Downloading LibTorch $VERSION..."
wget -q --show-progress "$URL" -O libtorch.zip
unzip -q libtorch.zip -d "$HOME"
rm libtorch.zip
echo "LibTorch $VERSION installed"

# 3. Install system dependencies – ignore PPA conflicts
echo "Installing system packages (ignoring non-critical conflicts)..."
sudo apt update -qq || true

sudo apt install -y --no-install-recommends \
    build-essential cmake ninja-build \
    libopencv-dev \
    libarrow-dev libparquet-dev \
    nlohmann-json3-dev \
    libgtest-dev \
    wget unzip ca-certificates || \
    echo "Some packages failed (usually due to unrelated PPAs – this is harmless)"

# 4. Export paths for this session
export CMAKE_PREFIX_PATH="$HOME/libtorch:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$HOME/libtorch/lib:$LD_LIBRARY_PATH"

echo ""
echo "Setup complete! You are ready."
echo "→ Run build:     ./build.sh clean && ./build.sh"
echo "→ Run training:  ./build/train"
