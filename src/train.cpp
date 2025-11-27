#include "dataset.h"
#include "act_policy.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdlib>
#include <filesystem>

int main() {
    namespace fs = std::filesystem;
    fs::create_directories("checkpoints");

    std::map<std::string, std::vector<float>> deltas{
        {"observation.image", {-0.033f, 0.0f}}
    };

    LeRobotDataset dataset("data/pusht", deltas, "observation.state", "action");
    dataset.print_all_column_names();
    std::cout << "Dataset loaded: " << dataset.size().value() << " frames\n";

    auto action_mean = dataset.get_action_mean().clone();
    auto action_std  = dataset.get_action_std().clone();
    auto state_mean  = dataset.get_state_mean().clone();
    auto state_std   = dataset.get_state_std().clone();

    std::cout << "State mean: " << state_mean << " std: " << state_std << "\n";
    std::cout << "Action mean: " << action_mean << " std: " << action_std << "\n";

    int state_dim  = dataset.get_state_mean().size(0);
    int action_dim = dataset.get_action_mean().size(0);
    int hidden_dim = 256;

    std::cout << "Using state_dim=" << state_dim 
              << " action_dim=" << action_dim 
              << " hidden_dim=" << hidden_dim << "\n";

    ACTPolicy policy(state_dim, action_dim, hidden_dim);
    policy->to(torch::kCPU);

    torch::optim::Adam optimizer(policy->parameters(), 1e-4);

    float total_loss = 0.0f;
    int log_interval = 10000;
    for (int step = 1; step <= 100000; ++step) {
        size_t idx = std::rand() % dataset.size().value();
        auto f = dataset.get(idx);

	// std::cout << "DEBUG FRAME " << idx << ":\n"
	// 	  << "  state raw = " << f.state << "\n"
	// 	  << "  action raw = " << f.action << "\n"
	// 	  << "  state norm = " << (f.state - state_mean) / (state_std + 1e-8) << "\n"
	// 	  << "  action norm = " << (f.action - action_mean) / (action_std + 1e-8) << "\n";
	// if (!f.images.empty()) {
	//   auto& img = f.images.begin()->second;
	//   std::cout << "  image size = " << img.size() << " type=" << img.type() << "\n";
	// }

        std::vector<cv::Mat> imgs;
        auto it = f.images.find(-0.066f);
	if (it != f.images.end()) imgs.push_back(it->second);
	it = f.images.find(0.0f);
	if (it != f.images.end()) imgs.push_back(it->second);

        // Fallback: 96x96 gray images (Push-T resolution)
        if (imgs.empty()) {
            imgs.emplace_back(96, 96, CV_8UC3, cv::Scalar(128,128,128));
        }
        if (imgs.size() == 1) {
            imgs.emplace_back(96, 96, CV_8UC3, cv::Scalar(128,128,128));
        }

        auto norm_state  = (f.state  - state_mean)  / (state_std  + 1e-5);
        auto norm_action = (f.action - action_mean) / (action_std + 1e-5);

        auto pred = policy->forward(imgs, norm_state);
        auto loss = torch::mse_loss(pred, norm_action);

	total_loss += loss.item<float>();
	if (step % 1000 == 0) {
	    std::cout << "Step: " << step << " | Loss: "  << loss.item<float>() << " | Action pred: " << pred.sizes() << "\n";
	}
        if (step % log_interval == 0) {
	    std::cout << "Avg Loss: " << (total_loss / log_interval) << "\n";
	    total_loss = 0.0f;
            torch::save(policy, "checkpoints/act_step" + std::to_string(step) + ".pt");
	}

        optimizer.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(policy->parameters(), 1.0);
        optimizer.step();

    }

    torch::save(policy, "checkpoints/act_final.pt");
    std::cout << "Training complete! Model saved to checkpoints/act_final.pt\n";
    return 0;
}
