# LeRobot-CPP
A C++ recreation of Hugging Face's LeRobot for robotics AI learning[](https://github.com/huggingface/lerobot)

## Dependencies
| Dep | Why | LeRobot Map | Resources |
|-----|-----|-------------|-----------|
| LibTorch | Tensors for states, actions, policies (e.g., ACT transformer) | PyTorch | [PyTorch C++ Docs](https://pytorch.org/cppdocs/) |
| Apache Arrow + Parquet | Parquet reads for hf_dataset (frames, episodes) | HF datasets (Arrow backend) | [Arrow C++ Docs](https://arrow.apache.org/docs/cpp/), [Quickstart](https://arrow.apache.org/docs/cpp/start.html) |
| OpenCV | Image/video for obs.images, viz | cv2 for videos/viz | [OpenCV Docs](https://docs.opencv.org/) |
| nlohmann/json | Parse meta.json (fps, episodes) | json/dataclasses | [JSON for Modern C++](https://github.com/nlohmann/json) |

## Setup
1. `source ./setup.sh` # Downloads LibTorch, installs apt deps (Arrow, OpenCV, JSON, GTest). Note: Source to set env.
2. `./build.sh` # CMake + make.
3. `./lerobot`  # Runs tests for all deps.

## Status
Work in progress. `main.cpp` tests deps (tensor, array, image, JSON).

