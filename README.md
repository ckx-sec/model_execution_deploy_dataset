# 模型部署执行环境 (Model Deploy Execution Environment)

本项目提供了一个 C++ 开发和执行环境，用于试验多种深度学习推理引擎，包括 **MNN, NCNN, ONNXRuntime, TNN, 和 TensorFlow Lite (TFLite)**。整个项目被设置为在 Docker 容器内构建，以确保一个一致且可复现的环境。

## 先决条件

- Docker

## 项目结构

-   `Dockerfile`: 定义了基于 `ubuntu:24.04` (ARM64) 的构建环境，包含了所有必需的编译器和工具。
-   `build.sh`: 用于编排所有构建任务的主脚本。
-   `CMakeLists.txt`: 项目的主 CMake 配置文件。
-   `src/`: 包含主库的源代码。
-   `examples/`: 包含每个推理后端的示例应用程序。
-   `cmake/`: 包含用于查找依赖项的辅助 CMake 模块。
-   `scripts/`: 包含用于下载和构建第三方库的脚本。
-   `third_party/`: 将包含最终编译好的依赖库和头文件。
-   `third_party_builds/`: 将包含下载的第三方库的源代码。
-   `assets/`: 用于存放模型文件和其他资源。

## 功能架构

本项目的核心是一个**用于 C++ 环境下多推理引擎部署、测试和比较的综合性工具集与资源库**。其架构并非一个统一的软件库，而是一个精心设计的“沙盒环境”，允许开发者直接使用不同引擎的原生API进行实验。


## 支持的模型与任务

本项目通过示例程序和资源文件，提供了对以下计算机视觉任务的支持：

| 任务类型 | 模型 | 支持格式 |
| :--- | :--- | :--- |
| 目标检测 | `yolov5_detector`, `ultraface_detector` | ONNX, NCNN, MNN, TFLite |
| 年龄估计 | `age_googlenet`, `ssrnet_age` | ONNX, NCNN, MNN, TFLite |
| 性别识别 | `gender_googlenet` | ONNX, NCNN, MNN, TFLite |
| 情绪识别 | `emotion_ferplus` | ONNX, NCNN, MNN, TFLite |
| 头部姿态估计 | `fsanet-1x1`, `fsanet-var` | ONNX, NCNN, MNN, TFLite |
| 人脸关键点检测 | `pfld_landmarks` | ONNX, NCNN, MNN, TFLite |

## 如何构建

所有的构建步骤都通过项目根目录下的 `./build.sh` 脚本进行管理。

### 控制构建选项

您可以通过设置环境变量来选择编译器和构建类型（优化等级）：

-   `CC`: 指定 C 编译器 (例如 `gcc`, `clang`)。
-   `CXX`: 指定 C++ 编译器 (例如 `g++`, `clang++`)。
-   `CMAKE_BUILD_TYPE`: 指定构建类型。
    -   `Release`: 用于生产发布，开启了高等级优化。
    -   `Debug`: 用于调试，包含了调试符号且关闭了优化。
    -   `RelWithDebInfo`: 发布版本带调试信息。

**示例:**

使用 `clang` 编译器并以 `Debug` 模式构建 MNN：
```bash
CC=clang CXX=clang++ CMAKE_BUILD_TYPE=Debug ./build.sh build-mnn
```

### 可用命令

-   `./build.sh build-docker`: 构建开发环境所需的 Docker 镜像。
-   `./build.sh prepare`: 克隆 MNN、NCNN、ONNXRuntime、TNN 和 TFLite 的源代码到 `third_party_builds/` 目录。
-   `./build.sh build-mnn`: 编译 MNN 库。
-   `./build.sh build-ncnn`: 编译 NCNN 库。
-   `./build.sh build-onnxruntime`: 编译 ONNXRuntime 库。
-   `./build.sh build-tnn`: 编译 TNN 库。
-   `./build.sh build-tflite`: 编译 TensorFlow Lite 库。
-   `./build.sh build-project`: 编译主项目库和示例。
-   `./build.sh all`: 按顺序运行所有必要步骤。
-   `./build.sh shell`: 进入构建容器内的交互式 `bash` shell，用于调试。
-   `./build.sh clean`: 删除项目的 `build/` 目录和已安装的 `third_party/` 库。

### 分步指南

要从头开始构建所有内容，请按照以下步骤操作：

1.  **构建 Docker 镜像**
    ```bash
    ./build.sh build-docker
    ```

2.  **下载依赖项**
    ```bash
    ./build.sh prepare
    ```

3.  **构建所有第三方库**
    ```bash
    ./build.sh build-mnn
    ./build.sh build-ncnn
    ./build.sh build-onnxruntime
    ./build.sh build-tnn
    ./build.sh build-tflite
    ```

4.  **构建项目**
    ```