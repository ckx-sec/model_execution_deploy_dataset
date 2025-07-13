#!/bin/bash
#
# Build script for ncnn for the new project.
# This script is intended to be run inside the builder Docker container.
#

set -e

# Respect the CMAKE_BUILD_TYPE environment variable, default to Release.
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

# Paths are relative to the project root inside the container (/app)
NCNN_SOURCE_DIR="/app/third_party_builds/ncnn"
OUTPUT_DIR="/app/third_party/ncnn"
BUILD_DIR="$NCNN_SOURCE_DIR/build"

# Prepare output directories
mkdir -p "$OUTPUT_DIR/lib"
mkdir -p "$OUTPUT_DIR/include"

echo "=> Building ncnn..."

cd "$NCNN_SOURCE_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# NCNN's install prefix behaves a bit differently, it creates an install subdir
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    -DNCNN_SHARED_LIB=ON \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DNCNN_VULKAN=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF

make -j$(nproc)
make install

echo "=> ncnn build complete."
echo "Libraries installed in $OUTPUT_DIR" 