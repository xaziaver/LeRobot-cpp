#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <torch/torch.h>
#include <torch/data/datasets/base.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

namespace fs = std::filesystem;

// represents a single data sample
struct Frame {
  // images at different delta timestamps
  std::map<float, cv::Mat> images;  // key = delta seconds

  torch::Tensor state;
  torch::Tensor action;
  int64_t episode_index = -1;
  int64_t frame_index = -1;
  double timestamp = 0.0;
};

class LeRobotDataset
  : public torch::data::datasets::Dataset<LeRobotDataset, Frame> {
public:
  explicit LeRobotDataset(const std::string& root_path,
			  const std::map<std::string, std::vector<float>>& delta_timestamps = {});

  // torch::data::Dataset API
  Frame get(size_t index) override;
  c10::optional<size_t> size() const override { return total_frames_; }

  //void print_column_names () const;
  size_t num_episodes() const { return episode_starts_.size(); }
  
 private:
  // --- Data ---
  std::vector<std::shared_ptr<arrow::Table>> tables_;     // one per chunk
  std::vector<size_t> chunk_frame_counts_;                // frames per chunk
  size_t total_frames_ = 0;

  // --- Episode indexing ---
  std::vector<size_t> episode_starts_;  // global index of first frame in episode

  // --- Video ---
  std::unordered_map<std::string, cv::VideoCapture> video_captures_;  // path → capture
  double fps_ = 30.0;

  // --- Delta timestamps ---
  std::map<std::string, std::vector<float>> delta_timestamps_;
  
  // --- Metadata ---
  nlohmann::json meta_;

  // --- Helpers ---
  void load_all_parquet(const fs::path& data_dir);
  void load_video(const fs::path& video_dir);
  void build_episode_index();
  cv::Mat decode_frame(const std::string& video_path, double timestamp_sec);
};
