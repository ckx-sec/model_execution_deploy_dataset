#!/bin/bash

# run_all_tests.sh
#
# A script to systematically test all executables in the build/bin directory.
# It intelligently maps executables to their required model and image assets.
#
# Usage:
# 1. Make sure you have built the project using ./build.sh
# 2. Run this script from the project root directory: ./run_all_tests.sh

# --- Configuration ---
# Assuming the script is run from the project root (model_deploy_execution_dataset)
BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/build/bin"
ASSETS_DIR="$BASE_DIR/assets"

# Array to store names of failed or skipped tests
failures=()

# --- Pre-run Checks ---
if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Directory $BIN_DIR not found."
    echo "Please build the project first with ./build.sh"
    exit 1
fi

if [ ! -d "$ASSETS_DIR" ]; then
    echo "Error: Assets directory $ASSETS_DIR not found."
    exit 1
fi


# --- Main Loop ---
# Iterate over all executables in the bin directory
for exe_path in "$BIN_DIR"/*; do
    # Skip if not an executable file
    if [ ! -x "$exe_path" ]; then
        continue
    fi

    exe_name=$(basename "$exe_path")
    echo "========================================================================"
    echo "INFO: Preparing to test: $exe_name"
    echo "========================================================================"

    # --- 1. Parse Executable Name ---
    model_base=""
    engine=""

    if [[ $exe_name == *"_tflite"* ]]; then
        engine="tflite"
        model_base=${exe_name%_tflite}
    elif [[ $exe_name == *"_onnxruntime"* ]]; then
        engine="onnxruntime"
        model_base=${exe_name%_onnxruntime}
    elif [[ $exe_name == *"_ncnn"* ]]; then
        engine="ncnn"
        model_base=${exe_name%_ncnn}
    elif [[ $exe_name == *"_mnn"* ]]; then
        engine="mnn"
        model_base=${exe_name%_mnn}
    else
        failures+=("SKIPPED: $exe_name (Could not determine engine)")
        continue
    fi

    # --- 2. Map to Asset Files ---
    # Based on source code analysis, map the base name to the actual asset files
    model_asset_name=""
    image_asset_name=""

    case $model_base in
        "age_googlenet")
            model_asset_name="age_googlenet"
            image_asset_name="test_lite_age_googlenet.jpg"
            ;;
        "emotion_ferplus")
            model_asset_name="emotion_ferplus"
            image_asset_name="test_lite_emotion_ferplus.jpg"
            ;;
        "fsanet_headpose")
            # This model requires two model files: a detector and a pose estimator.
            # We'll use ultraface_detector as the detector and fsanet-1x1 as the estimator.
            image_asset_name="test_lite_fsanet.jpg"
            ;;
        "gender_googlenet")
            model_asset_name="gender_googlenet"
            image_asset_name="test_lite_gender_googlenet.jpg"
            ;;
        "pfld_landmarks")
            model_asset_name="pfld_landmarks"
            image_asset_name="test_lite_face_landmarks_0.png"
            ;;
        "ssrnet_age")
            model_asset_name="ssrnet_age"
            image_asset_name="test_lite_ssrnet.jpg"
            ;;
        "ultraface_detector")
            model_asset_name="ultraface_detector"
            image_asset_name="test_lite_ultraface.jpg"
            ;;
        "yolov5_detector")
            model_asset_name="yolov5_detector"
            image_asset_name="test_lite_yolov5_1.jpg"
            ;;
        "mnist")
            # MNIST examples in the source code only require the model path.
            # However, the models are not present in the assets folder.
            failures+=("SKIPPED: $exe_name (MNIST models not in assets directory)")
            continue
            ;;
        *)
            failures+=("SKIPPED: $exe_name (No asset mapping found for model base '$model_base')")
            continue
            ;;
    esac

    # --- 3. Construct Arguments ---
    args=() # Initialize empty args array

    # Handle the special case for fsanet_headpose
    if [ "$model_base" = "fsanet_headpose" ]; then
        # The ncnn executable for fsanet now takes two models, similar to other engines.
        # It requires var.param, var.bin, 1x1.param, and 1x1.bin.
        if [ "$engine" = "ncnn" ]; then
            var_param_path="$ASSETS_DIR/fsanet-var.param"
            var_bin_path="$ASSETS_DIR/fsanet-var.bin"
            conv_param_path="$ASSETS_DIR/fsanet-1x1.param"
            conv_bin_path="$ASSETS_DIR/fsanet-1x1.bin"

            # Check if all model files exist
            if [ ! -f "$var_param_path" ] || [ ! -f "$var_bin_path" ] || \
               [ ! -f "$conv_param_path" ] || [ ! -f "$conv_bin_path" ]; then
                failures+=("ERROR: $exe_name (One or more NCNN models for 'fsanet_headpose' not found)")
                continue
            fi
            args=("$var_param_path" "$var_bin_path" "$conv_param_path" "$conv_bin_path")
        else
            var_model_path=""
            conv_model_path=""
            case $engine in
                "tflite")
                    var_model_path="$ASSETS_DIR/fsanet-var_float16.tflite"
                    conv_model_path="$ASSETS_DIR/fsanet-1x1_float16.tflite"
                    ;;
                "onnxruntime")
                    var_model_path="$ASSETS_DIR/fsanet-var.onnx"
                    conv_model_path="$ASSETS_DIR/fsanet-1x1.onnx"
                    ;;
                "mnn")
                    var_model_path="$ASSETS_DIR/fsanet-var.mnn"
                    conv_model_path="$ASSETS_DIR/fsanet-1x1.mnn"
                    ;;
            esac

            if [ ! -f "$var_model_path" ] || [ ! -f "$conv_model_path" ]; then
                failures+=("ERROR: $exe_name (One or more models for 'fsanet_headpose' not found)")
                continue
            fi
            args=("$var_model_path" "$conv_model_path")
        fi

    # Handle NCNN models which require param and bin paths
    elif [ "$engine" = "ncnn" ]; then
        param_path="$ASSETS_DIR/${model_asset_name}.param"
        bin_path="$ASSETS_DIR/${model_asset_name}.bin"

        if [ ! -f "$param_path" ] || [ ! -f "$bin_path" ]; then
            failures+=("ERROR: $exe_name (NCNN model files not found for '$model_asset_name')")
            continue
        fi
        args=("$param_path" "$bin_path")

    # For all other standard models
    else
        model_path=""
        case $engine in
            "tflite")
                # Default to float32 model, fallback to name without precision suffix
                model_path="$ASSETS_DIR/${model_asset_name}_float32.tflite"
                if [ ! -f "$model_path" ]; then
                    model_path="$ASSETS_DIR/${model_asset_name}.tflite"
                fi
                ;;
            "onnxruntime")
                model_path="$ASSETS_DIR/${model_asset_name}.onnx"
                ;;
            "mnn")
                model_path="$ASSETS_DIR/${model_asset_name}.mnn"
                ;;
        esac

        # Validate that the final model path exists
        if [ ! -f "$model_path" ]; then
            failures+=("ERROR: $exe_name (Model file not found for '$model_asset_name')")
            continue
        fi
        args=("$model_path")
    fi

    # Add image path as the last argument if one is required
    if [ -n "$image_asset_name" ]; then
        image_path="$ASSETS_DIR/$image_asset_name"
        if [ ! -f "$image_path" ]; then
            failures+=("ERROR: $exe_name (Image file '$image_asset_name' not found)")
            continue
        fi
        args+=("$image_path")
    fi

    # --- 4. Execute ---
    echo "RUNNING: $exe_path ${args[@]}"
    "$exe_path" "${args[@]}"
    echo # Newline for cleaner output

done

echo "========================================================================"
echo "INFO: All tests finished."
echo "========================================================================"

# --- Failure Summary ---
if [ ${#failures[@]} -ne 0 ]; then
    echo
    echo "--- Summary of Failures ---"
    for failure in "${failures[@]}"; do
        echo "- $failure"
    done
    echo "--------------------------"
fi 