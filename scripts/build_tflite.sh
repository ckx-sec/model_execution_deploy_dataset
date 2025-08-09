#!/bin/bash
set -e

# --- Configuration ---
# Upgraded to a recent stable version compatible with modern compilers
TENSORFLOW_VERSION="v2.16.1"
INSTALL_DIR="/app/third_party/tflite"
BUILD_DIR="/app/third_party_builds/tensorflow"

# --- Install a compatible compiler ---
# The default GCC 13 in the container is too new for TF's build system.
# We will install and use GCC 11, which is known to be compatible.
echo "--- Installing compatible compiler (GCC 11) ---"
apt-get update && apt-get install -y gcc-11 g++-11

echo "--- Building TensorFlow Lite ${TENSORFLOW_VERSION} ---"
echo "Source will be cloned to ${BUILD_DIR}"
echo "Installation will be in ${INSTALL_DIR}"

# --- Clone TensorFlow source if it doesn't exist ---
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Cloning TensorFlow repository..."
    git clone --depth 1 --branch "${TENSORFLOW_VERSION}" https://github.com/tensorflow/tensorflow.git "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"

# --- Build TFLite ---
rm -rf tflite_build
mkdir -p tflite_build

echo "Configuring CMake for TFLite..."
# Use the standard BUILD_SHARED_LIBS flag to build a .so file
CC=gcc-11 CXX=g++-11 cmake -S tensorflow/lite -B tflite_build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DBUILD_SHARED_LIBS=ON

echo "Building TFLite..."
cmake --build tflite_build -j$(nproc)

# --- Manually Install Artifacts ---
echo "Creating final destination directories..."
mkdir -p "${INSTALL_DIR}/lib"
mkdir -p "${INSTALL_DIR}/include"

echo "Copying library and headers..."
# Copy the compiled shared library
cp tflite_build/libtensorflow-lite.so "${INSTALL_DIR}/lib/"
# Copy the necessary headers from the source directory
cp -r tensorflow/lite "${INSTALL_DIR}/include/tensorflow"
# TFLite headers depend on flatbuffers headers, so we must copy them too.
# They are downloaded by the TFLite build process into the build directory.
echo "Copying flatbuffers headers..."
cp -r tflite_build/flatbuffers/include/flatbuffers "${INSTALL_DIR}/include/"

echo "--- TensorFlow Lite build and installation complete ---"
echo "Library and headers are in ${INSTALL_DIR}" 