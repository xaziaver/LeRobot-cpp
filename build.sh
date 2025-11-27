#!/bin/bash
set -e

BUILD_DIR="build"
if [ "$1" == "clean" ]; then
    rm -rf "$BUILD_DIR"
    echo "Cleaned build directory"
    exit 0
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_PREFIX_PATH="$HOME/libtorch" \
      -DTorch_DIR="$HOME/libtorch/share/cmake/Torch" \
      -DCMAKE_BUILD_TYPE=Release \
      ..

cmake --build . --config Release -j$(nproc)

echo ""
echo "Build successful!"
echo "Run: build/train"
echo "Or with libraries: LD_LIBRARY_PATH=$HOME/libtorch/lib build/train"
