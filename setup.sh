#!/bin/bash
set -e  # exit on error

LIBTORCH_DIR="$HOME/libtorch"
VERSION="latest"
URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch..."
    wget -O libtorch.zip "$URL"
    unzip libtorch.zip -d "$LIBTORCH_DIR"
    rm libtorch.zip
    echo "LibTorch installed at $LIBTORCH_DIR"
else
    echo "LibTorch already at $LIBTORCH_DIR"
fi

export CMAKE_PREFIX_PATH="$LIBTORCH_DIR"
echo "Set CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
