#include "act_policy.h"

ACTPolicyImpl::ACTPolicyImpl(int state_dim, int action_dim, int hidden)
    : hidden_dim(hidden) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(4)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 5).stride(2)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, hidden, 3).stride(2)));
    
    state_proj = register_module("state_proj", torch::nn::Linear(state_dim, hidden));
    head = register_module("head", torch::nn::Linear(hidden, action_dim));

    auto layer = torch::nn::TransformerEncoderLayer(
        torch::nn::TransformerEncoderLayerOptions(hidden, 8).dropout(0.1));
    encoder = register_module("encoder", torch::nn::TransformerEncoder(layer, 4));

    // Safe initialization
    float scale = std::sqrt(1.0f / std::max(state_dim, 1));
    torch::nn::init::uniform_(state_proj->weight, -scale, scale);
    torch::nn::init::constant_(state_proj->bias, 0);
    torch::nn::init::kaiming_normal_(head->weight, 0.0, torch::kFanIn, torch::kReLU);
    torch::nn::init::constant_(head->bias, 0);
}

torch::Tensor ACTPolicyImpl::forward(const std::vector<cv::Mat>& images,
                                     const torch::Tensor& state) {
    torch::Tensor img_tokens;

    if (!images.empty() && !images.back().empty()) {
        auto& img = images.back();
        auto x = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8)
                     .to(torch::kFloat32) / 255.0;
        x = x.permute({2, 0, 1}).unsqueeze(0);

        x = torch::relu(conv1(x));
        x = torch::relu(conv2(x));
        x = torch::relu(conv3(x));  // ← now outputs hidden_dim channels
        x = torch::adaptive_avg_pool2d(x, {7, 7});
        x = x.flatten(2).transpose(1, 2);
        img_tokens = x.squeeze(0);  // [49, hidden_dim]
    } else {
        img_tokens = torch::zeros({49, hidden_dim}, torch::kFloat32);
    }

    auto state_token = state_proj(state).unsqueeze(0);  // [1, hidden_dim]

    auto seq = torch::cat({img_tokens, state_token}, 0);  // [50, hidden_dim]
    seq = seq.unsqueeze(1);                               // [50, 1, hidden_dim]

    auto encoded = encoder(seq);
    encoded = encoded.squeeze(1);

    return head(encoded.index({-1}));  // [action_dim]
}
