#!/bin/bash
set -e  # exit on error

# LibTorch
LIBTORCH_DIR="$HOME/libtorch"
VERSION="latest"#2.4.1
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

# Arrow
echo "Installing Arrow (Pop!_OS workaround)..."
sudo apt update
sudo apt install -y -V ca-certificates lsb-release wget
DISTRO_ID="ubuntu"
CODENAME=$(lsb_release --codename --short)
wget -O /tmp/apache-arrow-apt-source-latest-${CODENAME}.deb "https://packages.apache.org/artifactory/arrow/${DISTRO_ID}/apache-arrow-apt-source-latest-${CODENAME}.deb"
chmod 644 /tmp/apache-arrow-apt-source-latest-${CODENAME}.deb
sudo apt install -y -V  /tmp/apache-arrow-apt-source-latest-${CODENAME}.deb
sudo apt update
sudo apt install -y -V libarrow-dev libparquet-dev
rm /tmp/apache-arrow-apt-source-latest-${CODENAME}.deb

# Other deps
sudo apt install -y -V libopencv-dev nlohmann-json3-dev libgtest-dev

echo "Setup complete! Run 'source ./setup.sh' then './build.sh'"
