# LeRobot-CPP
A C++ recreation of Hugging Face's [LeRobot](https://github.com/huggingface/lerobot) for robotics AI learning.

This project recreates key components like dataset loading (Apache Arrow/Parquet), normalization, and a basic ACT (Action Chunking Transformer) policy in pure C++ with LibTorch. It supports training on datasets like Push-T, with plans for full ACT chunking, simulation testing, and hardware deployment.

## Dependencies
| Dep | Why | Resources |
|-----|-----|-----------|
| LibTorch | Tensors for states, actions, policies (e.g., ACT transformer) | [PyTorch C++ Docs](https://pytorch.org/cppdocs/) |
| Apache Arrow + Parquet | Parquet reads for hf_dataset (frames, episodes) | [Arrow C++ Docs](https://arrow.apache.org/docs/cpp/), [Quickstart](https://arrow.apache.org/docs/cpp/start.html) |
| OpenCV | Image/video for obs.images, viz | [OpenCV Docs](https://docs.opencv.org/) |
| nlohmann/json | Parse meta.json (fps, episodes) | [JSON for Modern C++](https://github.com/nlohmann/json) |

## Setup
1. `source ./setup.sh` # Downloads LibTorch, installs apt deps (Arrow, OpenCV, JSON, GTest). Note: Source to set env.
2. `./build.sh` # CMake + make.
3. `LD_LIBRARY_PATH=/path/to/libtorch/lib build/train` # Runs training on Push-T (or other datasets).

## Current Progress
- **Dataset Loading**: Supports LeRobot-style datasets (e.g., Push-T). Normalization stats (mean/std) computed/cached.
- **Policy**: Basic ACT policy implemented with CNN (image feat), state projection, Transformer encoder, and linear head. Predicts single actions; trained via MSE on normalized data.
- **Training**: Single-sample SGD with Adam, grad clipping. Logs loss every 1k steps, avg every 10k. Tested on Push-T (25k frames, state/action dim=2).
- **Tweaks**: Configurable hidden_dim (64/256), lower LR for stability.


## Status
Work in progress. `train.cpp` trains the policy; add `inference.cpp` for deployment. Tests in `main.cpp` verify deps (tensor, array, image, JSON).

## Future Work
- Implement full ACT chunking (predict multiple future actions for planning).
- Add simulation (e.g., simple 2D Push-T in C++)
- Hardware support (e.g., SO-101 arm via serial SDK).
- Multi-dataset training (e.g., ALOHA, xArm).
