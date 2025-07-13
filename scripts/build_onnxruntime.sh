#!/bin/bash
#
# Build script for ONNX Runtime for the new project.
# This script is intended to be run inside the builder Docker container.
#

set -e

# Respect the CMAKE_BUILD_TYPE environment variable, default to Release.
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

# Paths are relative to the project root inside the container (/app)
ORT_SOURCE_DIR="/app/third_party_builds/onnxruntime"
OUTPUT_DIR="/app/third_party/onnxruntime"
BUILD_DIR="$ORT_SOURCE_DIR/build"

# Prepare output directories
mkdir -p "$OUTPUT_DIR/lib"
mkdir -p "$OUTPUT_DIR/include"

echo "=> Building ONNX Runtime..."

cd "$ORT_SOURCE_DIR"
rm -rf "$BUILD_DIR"

# ONNX Runtime's build script
# The --use_openmp flag is not recognized in recent versions, it's often enabled by default on Linux.
./build.sh --config ${CMAKE_BUILD_TYPE} --build_shared_lib --parallel --skip_tests \
    --allow_running_as_root \
    --cmake_extra_defines CMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    --build_dir "$BUILD_DIR"

# The build script doesn't always run install, so we do it manually
echo "=> Installing ONNX Runtime..."
cmake --build "$BUILD_DIR/${CMAKE_BUILD_TYPE}" --target install

echo "=> ONNX Runtime build complete."
echo "Libraries installed in $OUTPUT_DIR" 