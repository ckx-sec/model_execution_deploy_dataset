#!/bin/bash
#
# This script clones the required ML libraries from their repositories.
#

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT="$SCRIPT_DIR/.."
DEPS_DIR="$PROJECT_ROOT/third_party_builds"

mkdir -p "$DEPS_DIR"
cd "$DEPS_DIR"

echo "Cloning dependencies into $DEPS_DIR"

if [ ! -d "MNN" ]; then
    git clone https://github.com/alibaba/MNN.git
fi

if [ ! -d "ncnn" ]; then
    git clone https://github.com/Tencent/ncnn.git
fi

if [ ! -d "onnxruntime" ]; then
    git clone https://github.com/microsoft/onnxruntime.git
fi

if [ ! -d "TNN" ]; then
    git clone https://github.com/Tencent/TNN.git
fi

echo "All dependencies are cloned." 