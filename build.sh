#!/bin/bash
set -e
BUILD_DIR="build"

if [ "$1" == "clean" ]; then
    rm -rf "$BUILD_DIR"
    echo "Cleaned build dir"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..  # Hardcode for now
cmake --build . --config Release -j4
echo "Build done! Run ./build/lerobot"
