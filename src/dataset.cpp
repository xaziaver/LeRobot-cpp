#include "dataset.h"
#include <iostream>

using json = nlohmann::json;

// helper: safe tensor from dense FloatArray (PushT uses fixed length)
static torch::Tensor dense_float_to_tensor(const std::shared_ptr<arrow::Array>& col,
					   int64_t row,
					   int64_t expected_len = 14) {
  if (!col || col->length() <= row) return torch::zeros({expected_len});

  auto float_arr = std::static_pointer_cast<arrow::FloatArray>(col);
  if (!float_arr) {
    std::cerr << "WARNING: Column not FloatArray!\n";
    return torch::zeros({expected_len});
  }
  const float* data = float_arr->raw_values() + row * expected_len;
  return torch::from_blob(const_cast<float*>(data),
			  {expected_len},
			  torch::kFloat32).clone();
}

// constructor
LeRobotDataset::LeRobotDataset(const std::string& root_path,
			       const std::map<std::string, std::vector<float>>& delta_timestamps)
  : delta_timestamps_(delta_timestamps) {
  fs::path root(root_path);

  // 1. load all parquet chunks
  load_all_parquet(root / "data");

  // 2. load video (one per camera)
  load_video(root / "videos");

  // 3. build episode index
  build_episode_index();

  // 4. Load meta
  fs::path meta_path = fs::path(root_path) / "meta" / "info.json";
  if (fs::exists(meta_path)) {
      std::ifstream f(meta_path);
      f >> meta_;
      if (meta_.contains("fps")) fps_ = meta_["fps"];
  }
}

// load every parquet file in data/*/chunk-*/
void LeRobotDataset::load_all_parquet(const fs::path& data_dir) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  std::unique_ptr<parquet::arrow::FileReader> reader;
  std::shared_ptr<arrow::Table> table;
  for (const auto& chunk_dir : fs::directory_iterator(data_dir)) {
    if (!chunk_dir.is_directory()) continue;
    for (const auto& file : fs::directory_iterator(chunk_dir)) {
      if (file.path().extension() != ".parquet") continue;

      // open file
      auto file_result = arrow::io::ReadableFile::Open(file.path().string());
      if (!file_result.ok()) {
        throw std::runtime_error("Failed to open parquet file: " +
                                 file_result.status().ToString());
      }
      infile = *file_result;
      // create arrow reader
      auto reader_status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
      if (!reader_status.ok()) {
        throw std::runtime_error("Failed to create parquet reader: " +
                                 reader_status.status().ToString());
      }
      reader = std::move(*reader_status);
      // load into table
      PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
      tables_.push_back(table);
      chunk_frame_counts_.push_back(table->num_rows());
      total_frames_ += table->num_rows();
    }
  }
  std::cout << "Loaded " << tables_.size() << " chunks, " << total_frames_ << " frames\n";
}

// load video files
void LeRobotDataset::load_video(const fs::path& video_dir) {
  for (const auto& cam_dir : fs::directory_iterator(video_dir)) {
    if (!cam_dir.is_directory()) continue;
    for (const auto& chunk_dir : fs::directory_iterator(cam_dir)) {
      if (!chunk_dir.is_directory()) continue;
      for (const auto& file : fs::directory_iterator(chunk_dir)) {
        if (file.path().extension() != ".mp4") continue;
        std::string path = file.path().string();
        video_captures_[path].open(path);
        if (!video_captures_[path].isOpened())
          std::cerr << "Failed to open video: " << path << "\n";
      }
    }
  }
}

// build episode start indices
void LeRobotDataset::build_episode_index() {
    size_t global_idx = 0;
    for (const auto& table : tables_) {
        auto ep_col = table->GetColumnByName("episode_index");
        if (!ep_col) continue;
        auto arr = std::static_pointer_cast<arrow::Int64Array>(ep_col->chunk(0));
        int64_t prev = -1;
        for (int64_t i = 0; i < arr->length(); ++i) {
            int64_t ep = arr->Value(i);
            if (ep != prev && i > 0) {
                episode_starts_.push_back(global_idx + i);
            }
            prev = ep;
        }
        global_idx += table->num_rows();
    }
    episode_starts_.push_back(0);  // first episode
    std::sort(episode_starts_.begin(), episode_starts_.end());
}

// decode image at timestamp
cv::Mat LeRobotDataset::decode_frame(const std::string& video_path, double timestamp_sec) {
    auto& cap = video_captures_[video_path];
    if (!cap.isOpened()) return cv::Mat();

    int frame_idx = static_cast<int>(timestamp_sec * fps_);
    cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
    cv::Mat frame;
    cap >> frame;
    return frame;
}

// get() - full frame with delta images
Frame LeRobotDataset::get(size_t global_index) {
    // Find which chunk
    size_t chunk_idx = 0, local_idx = global_index;
    for (size_t i = 0; i < chunk_frame_counts_.size(); ++i) {
        if (local_idx < chunk_frame_counts_[i]) {
            chunk_idx = i;
            break;
        }
        local_idx -= chunk_frame_counts_[i];
    }
    auto table = tables_[chunk_idx];

    Frame f;
    f.state = dense_float_to_tensor(table->GetColumnByName("observation.state")->chunk(0), local_idx);
    f.action = dense_float_to_tensor(table->GetColumnByName("action")->chunk(0), local_idx);

    // Timestamp
    auto ts_arr = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("timestamp")->chunk(0));
    f.timestamp = ts_arr->Value(local_idx);

    // Video path (assume one camera)
    std::string video_path = (fs::path(table->schema()->metadata()->Get("video_path").ValueOr("")).string());
    if (video_path.empty()) {
        // fallback: first video
        if (!video_captures_.empty()) video_path = video_captures_.begin()->first;
    }

    // Delta images
    for (const auto& [modality, deltas] : delta_timestamps_) {
        if (modality != "observation.image") continue;
        for (float delta : deltas) {
            double target_ts = f.timestamp + delta;
            if (target_ts < 0) continue;
            cv::Mat img = decode_frame(video_path, target_ts);
            if (!img.empty()) f.images[delta] = img;
        }
    }

    return f;
}


//  print_column_names()
//void LeRobotDataset::print_column_names() const {
//    for (int i = 0; i < table_->num_columns(); ++i) {
//        std::cout << table_->ColumnNames()[i] << "\t";
//    }
//    std::cout << "\n";
//}
