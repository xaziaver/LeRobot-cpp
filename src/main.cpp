#include "dataset.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>

void run_tests() {
  // test 1: LibTorch
  // tensors for policy/state in LeRobot
  std::cout << "Testing LibTorch...\n";
  // mock state
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "Tensor (state/action): " << tensor << std::endl;

  // define & interact with modules
  // neural networks composed of modules
  // base module: torch::nn::Module
  // module usually contains three kinds of sub-objects:
  // parameters, buffers and submodules (nested module)
  // parameters(e.g. trainable weights) /
  // buffers(e.g. means and variances for batch normalization)
  // store state in form of tensors
  // parameters record gradients, while buffers do not

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M)
      // create submodule by adding to initializer list
      : linear(register_module("linear", torch::nn::Linear(N,M))) {
      another_bias = register_parameter("b", torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor input) {
      return linear(input) + another_bias;
    }
    torch::nn::Linear linear;
    torch::Tensor another_bias;
  };

  Net net = Net(3, 2);
  std::cout << net << " forward(): " << net.forward(tensor) << std::endl;

  std::cout << "iterating through the module tree's parameters\n";
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }

  // Test 2: Arrow
  // dataset frames, LeRobot's hf_dataset
  std::cout << "\nTesting Arrow...\n";
  arrow::Int32Builder builder;
  // mock frame index
  builder.Append(42);
  std::shared_ptr<arrow::Array> arr;
  builder.Finish(&arr);
  std::cout << "Arrow array length: " << arr->length() << std::endl;

  // Test 3: OpenCV
  // image for obs.images
  std::cout << "\nTesting OpenCV...\n";
  cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0, 255, 0));
  // mock camera frame
  cv::imwrite("test.png", img);
  std::cout << "Wrote test.png\n";

  // Test 4: nlohmann/json
  // metadata for episodes
  std::cout << "\nTesting JSON...\n";
  // mock LeRobot meta
  nlohmann::json meta = {{"fps", 30}, {"robot_type", "arm"}};
  std::cout << "JSON meta: " << meta.dump(2) << std::endl;

}

int main() {
  run_tests();
  std::map<std::string, std::vector<float>> deltas = {
    {"observation.image", {-0.1, 0.0}}
  };
  LeRobotDataset ds("data/pusht", deltas);
  int frame_num = 100;
  Frame f = ds.get(frame_num);
  std::cout << "Frame " << frame_num << " state = " << f.state << "\n";
  cv::imwrite("past.png", f.images[-0.1]);
  cv::imwrite("now.png", f.images[0.0]);
}

