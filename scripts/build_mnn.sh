#!/bin/bash
#
# Build script for MNN for the new project.
# This script is intended to be run inside the builder Docker container.
#

set -e

# Respect the CMAKE_BUILD_TYPE environment variable, default to Release.
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

# Paths are relative to the project root inside the container (/app)
MNN_SOURCE_DIR="/app/third_party_builds/MNN"
OUTPUT_DIR="/app/third_party/mnn"
BUILD_DIR="$MNN_SOURCE_DIR/build"

# Prepare output directories
mkdir -p "$OUTPUT_DIR/lib"
mkdir -p "$OUTPUT_DIR/include"

echo "=> Building MNN..."

cd "$MNN_SOURCE_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DMNN_BUILD_DEMO=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_BUILD_CONVERTER=OFF

make -j$(nproc)
make install

echo "=> MNN build complete."
echo "Libraries installed in $OUTPUT_DIR" 