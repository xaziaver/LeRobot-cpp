#pragma once
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

namespace fs = std::filesystem;

struct Frame {
    std::map<float, cv::Mat> images;
    torch::Tensor state;
    torch::Tensor action;
    double timestamp = 0.0;
};

class LeRobotDataset : public torch::data::datasets::Dataset<LeRobotDataset, Frame> {
public:
    LeRobotDataset(const std::string& root_path,
                   const std::map<std::string, std::vector<float>>& delta_timestamps = {},
		   const std::string& state_col = "observation.state",
		   const std::string& action_col = "action")
        : delta_timestamps_(delta_timestamps),
	  state_column_name_(state_col),
	  action_column_name_(action_col),
          ACTION_MEAN(torch::zeros({2}, torch::kFloat32)),
          ACTION_STD(torch::ones({2}, torch::kFloat32)),
          STATE_MEAN(torch::zeros({2}, torch::kFloat32)),
          STATE_STD(torch::ones({2}, torch::kFloat32)) {
        fs::path root(root_path);
        load_all_parquet(root / "data");
        load_video(root / "videos");
        build_episode_index();

        fs::path meta_path = fs::path(root_path) / "meta" / "info.json";
        if (fs::exists(meta_path)) {
            std::ifstream f(meta_path);
            f >> meta_;
            if (meta_.contains("fps")) fps_ = meta_["fps"];
        }
        load_normalization_stats();
    }

    Frame get(size_t index) override;
    void print_all_column_names() const;
    c10::optional<size_t> size() const override { return total_frames_; }
    void set_load_images(bool enable) { load_images_ = enable; } 
    torch::Tensor get_action_mean() const { return ACTION_MEAN; }
    torch::Tensor get_action_std()  const { return ACTION_STD; }
    torch::Tensor get_state_mean()  const { return STATE_MEAN; }
    torch::Tensor get_state_std()   const { return STATE_STD; }

private:
    bool load_images_ = true;
    std::string state_column_name_;
    std::string action_column_name_;
    std::vector<std::shared_ptr<arrow::Table>> tables_;
    std::vector<size_t> chunk_frame_counts_;
    std::vector<size_t> episode_starts_;
    size_t total_frames_ = 0;
    std::unordered_map<std::string, cv::VideoCapture> video_captures_;
    double fps_ = 30.0;
    std::map<std::string, std::vector<float>> delta_timestamps_;
    nlohmann::json meta_;

    torch::Tensor ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD;

    void compute_normalization_from_arrow();
    void load_normalization_stats();
    void load_all_parquet(const fs::path& data_dir);
    void load_video(const fs::path& video_dir);
    void build_episode_index();
    cv::Mat decode_frame(const std::string& video_path, double timestamp_sec);
};
