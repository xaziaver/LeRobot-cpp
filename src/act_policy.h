#pragma once
#include <torch/torch.h>
#include <torch/nn/module.h>
#include "dataset.h"

struct ACTPolicyImpl : torch::nn::Module {
    ACTPolicyImpl(int state_dim, int action_dim, int hidden = 256);
    torch::Tensor forward(const std::vector<cv::Mat>& images, const torch::Tensor& state);

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear state_proj{nullptr};
    torch::nn::TransformerEncoder encoder{nullptr};
    torch::nn::Linear head{nullptr};
    int hidden_dim = 256;  // Store hidden dim
};

TORCH_MODULE(ACTPolicy);
