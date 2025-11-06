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

namespace fs = std::filesystem;

// represents a single data sample
struct Frame {
  // images(s) - for now keep only the *current* frame
  // later add a std::map<float,cv::Mat> for delta timestamps
  cv::Mat image;

  // state/action are plan float tensors (LeRobot uses float32)
  torch::Tensor state;
  torch::Tensor action;
};

// LeRobotDataset encapsulates 3 main files:
// (1) hf_dataset, to read any value from parquet files
// (2) metadata:
//  - info: various info about dataset like shapes, keys, fps
//  - stats: dataset statistics of modalities for normalization
//  - tasks: prompts for each task of the dataset (task-conditioned training)
// (3) videos (optional) from which frames are loaded to be synchronous with data
//     from parquet files
class LeRobotDataset : public
torch::data::datasets::Dataset<LeRobotDataset, Frame> {
  // initialize data: two options
  // (1) dataset exists on local disk
  // (2) dataset on Hugging Face Hub at https://huggingface.co/datasets/{repo_id}

  // EXAMPLE FILE: at ~/Code/LeRobot-cpp/data/pusht
  // files downloaded directly, file structure is:
  //  pusht
  //      ├── data
  //      │   └── chunk-000
  //      │       └── file-000.parquet
  //      ├── meta
  //      │   ├── episodes
  //      │   │   └── chunk-000
  //      │   │       └── file-000.parquet
  //      │   ├── info.json
  //      │   ├── stats.json
  //      │   └── tasks.parquet
  //      └── videos
  //          └── observation.image
  //              └── chunk-000
  //                  └── file-000.mp4
public:
  // constructor - loads the *single* parquet file that lives under
  // <root>/data/chunk-000/file-000.parquet
  explicit LeRobotDataset(const std::string& root_path);

  // torch::data::Dataset API
  Frame get(size_t index) override;
  c10::optional<size_t> size() const override { return table_->num_rows(); }

  // Helper utilities
  void print_column_names () const;
  const std::shared_ptr<arrow::Table>& arrow_table() const { return table_; }
  
 private:
  std::shared_ptr<arrow::Table> table_;     // the whole parquet fi
  fs::path parquet_path_;                   // full path to the .parquet file
  nlohmann::json meta_;                     // meta/info.json
};
