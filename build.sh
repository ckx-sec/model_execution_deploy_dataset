#!/bin/bash
#
# Main build script for the project.
#
set -e

# --- Configuration ---
# You can override these by setting them before calling the script.
# Example: CC=clang CXX=clang++ CMAKE_BUILD_TYPE=Debug ./build.sh all
CC=${CC:-gcc}
CXX=${CXX:-g++}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
DOCKER_IMAGE="mei-execution-builder"
PROJECT_ROOT=$(cd $(dirname $0); pwd)

# --- Helper Functions ---
print_usage() {
    echo "Usage: $0 {build-docker|prepare|build-mnn|build-ncnn|build-onnxruntime|build-tnn|build-project|all|shell|clean}"
    echo ""
    echo "You can control the compiler and build type using environment variables:"
    echo "  CC: C compiler (e.g., gcc, clang)"
    echo "  CXX: C++ compiler (e.g., g++, clang++)"
    echo "  CMAKE_BUILD_TYPE: Build type (e.g., Release, Debug, RelWithDebInfo)"
    echo "Example: CMAKE_BUILD_TYPE=Debug ./build.sh build-mnn"
}

run_in_docker() {
    echo "==> Running in Docker with CC=${CC}, CXX=${CXX}, CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    docker run --rm \
        -e CC="${CC}" \
        -e CXX="${CXX}" \
        -e CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -v "$PROJECT_ROOT:/app" -w /app "$DOCKER_IMAGE" bash -c "$1"
}

# --- Commands ---
cmd_build_docker() {
    echo "=> Building Docker image: $DOCKER_IMAGE..."
    docker build -t "$DOCKER_IMAGE" .
}

cmd_prepare() {
    echo "=> Preparing dependencies..."
    bash "$PROJECT_ROOT/scripts/prepare_dependencies.sh"
}

cmd_build_mnn() {
    echo "=> Building MNN..."
    run_in_docker "./scripts/build_mnn.sh"
}

cmd_build_ncnn() {
    echo "=> Building NCNN..."
    run_in_docker "./scripts/build_ncnn.sh"
}

cmd_build_onnxruntime() {
    echo "=> Building ONNXRuntime..."
    run_in_docker "./scripts/build_onnxruntime.sh"
}

cmd_build_tnn() {
    echo "=> Building TNN..."
    run_in_docker "./scripts/build_tnn.sh"
}

cmd_build_project() {
    echo "=> Building main project..."
    local build_cmd="
        rm -rf build && mkdir build && cd build &&
        cmake .. -DCMAKE_BUILD_TYPE=\${CMAKE_BUILD_TYPE} &&
        make -j\$(nproc)
    "
    run_in_docker "$build_cmd"
}

cmd_all() {
    cmd_build_docker
    cmd_prepare
    cmd_build_mnn
    cmd_build_ncnn
    cmd_build_onnxruntime
    cmd_build_tnn
    cmd_build_project
    echo "=> All tasks completed."
}

cmd_shell() {
    echo "=> Entering Docker container shell..."
    docker run --rm -it -v "$PROJECT_ROOT:/app" -w /app "$DOCKER_IMAGE" bash
}

cmd_clean() {
    echo "=> Cleaning build artifacts..."
    rm -rf "$PROJECT_ROOT/build"
    rm -rf "$PROJECT_ROOT/third_party"
    # Keep the downloaded source code in third_party_builds
}


# --- Main Logic ---
if [ "$#" -ne 1 ]; then
    print_usage
    exit 1
fi

case "$1" in
    build-docker) cmd_build_docker ;;
    prepare) cmd_prepare ;;
    build-mnn) cmd_build_mnn ;;
    build-ncnn) cmd_build_ncnn ;;
    build-onnxruntime) cmd_build_onnxruntime ;;
    build-tnn) cmd_build_tnn ;;
    build-project) cmd_build_project ;;
    all) cmd_all ;;
    shell) cmd_shell ;;
    clean) cmd_clean ;;
    *) print_usage; exit 1 ;;
esac 