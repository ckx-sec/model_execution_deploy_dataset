#!/bin/bash
#
# Build script for TNN for the new project.
# This script is intended to be run inside the builder Docker container.
#

set -e

# Respect the CMAKE_BUILD_TYPE environment variable, default to Release.
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

# Paths are relative to the project root inside the container (/app)
TNN_SOURCE_DIR="/app/third_party_builds/TNN"
OUTPUT_DIR="/app/third_party/tnn"
BUILD_DIR="$TNN_SOURCE_DIR/build"

# Prepare output directories
mkdir -p "$OUTPUT_DIR/lib"
mkdir -p "$OUTPUT_DIR/include"

# TNN needs its own build script to be run
# We will create a script to run inside the container
echo "=> Building TNN..."

cd "$TNN_SOURCE_DIR"

# Fix for missing <cstdint> include in TNN source for newer compilers
echo "=> Applying patch for TNN build..."
sed -i '17i#include <cstdint>' source/tnn/utils/data_type_utils.cc

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# TNN has a different build system
# For ARM Linux, we can use the provided script
# But here we use cmake for better control
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    -DTNN_BUILD_SHARED=ON \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_X86_ENABLE=OFF \
    -DTNN_ARM_ENABLE=ON \
    -DTNN_OPENCL_ENABLE=OFF \
    -DTNN_METAL_ENABLE=OFF \
    -DTNN_TEST_ENABLE=OFF \
    -DTNN_UNIT_TEST_ENABLE=OFF

make -j$(nproc)
# TNN does not have a standard install target, so we copy files manually
find . -name "libTNN.so" -exec cp {} "$OUTPUT_DIR/lib/" \;
cp -r ../include/* "$OUTPUT_DIR/include/"

echo "=> TNN build complete."
echo "Libraries installed in $OUTPUT_DIR" 