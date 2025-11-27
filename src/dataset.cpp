#include "dataset.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using json = nlohmann::json;

static const fs::path NORM_CACHE_PATH = fs::path("/tmp") / "lerobot_norm_cache.json";

static torch::Tensor read_fsl_tensor(
    const std::shared_ptr<arrow::Array>& array,
    int64_t row,
    int expected_dim)
{
    if (!array || row >= array->length() || array->IsNull(row)) {
        return torch::zeros({expected_dim}, torch::kFloat32);
    }

    if (auto fsl = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(array)) {
        auto values = std::static_pointer_cast<arrow::FloatArray>(fsl->values());
        int offset = static_cast<int>(row) * expected_dim;
        return torch::from_blob(
            const_cast<float*>(values->raw_values() + offset),
            {expected_dim},
            torch::kFloat32
        ).clone();
    }

    // Fallback
    if (auto fa = std::dynamic_pointer_cast<arrow::FloatArray>(array)) {
        return torch::from_blob(
            const_cast<float*>(fa->raw_values() + row),
            {expected_dim},
            torch::kFloat32
        ).clone();
    }

    return torch::zeros({expected_dim}, torch::kFloat32);
}

void LeRobotDataset::compute_normalization_from_arrow() {
    std::cout << "Computing normalization stats from Arrow tables (correct, safe)...\n";

    int state_dim = -1;
    int action_dim = -1;

    torch::Tensor state_sum   = torch::zeros({32}, torch::kFloat64);  // big enough
    torch::Tensor state_sum_sq = torch::zeros({32}, torch::kFloat64);
    torch::Tensor action_sum   = torch::zeros({32}, torch::kFloat64);
    torch::Tensor action_sum_sq = torch::zeros({32}, torch::kFloat64);

    int64_t valid_count = 0;

    for (const auto& table : tables_) {
        auto state_col  = table->GetColumnByName(state_column_name_);
        auto action_col = table->GetColumnByName(action_column_name_);
        if (!state_col || !action_col) continue;

        // Arrow tables can have multiple chunks → iterate over all chunks
        for (int c = 0; c < state_col->num_chunks(); ++c) {
            auto state_chunk  = state_col->chunk(c);
            auto action_chunk = action_col->chunk(c);

            // Helper lambda for fixed-size-list → flat float vector
            auto read_fsl = [](const std::shared_ptr<arrow::Array>& arr, int& dim)
                -> std::vector<float> {
                if (auto fsl = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(arr)) {
                    if (dim == -1) dim = fsl->list_type()->list_size();
                    auto values = std::static_pointer_cast<arrow::FloatArray>(fsl->values());
                    return std::vector<float>(values->raw_values(),
                                             values->raw_values() + values->length());
                }
                return {};
            };

            // Try FixedSizeListArray first (most common in LeRobot)
            std::vector<float> all_state_vals;
            std::vector<float> all_action_vals;
            int chunk_state_dim = -1, chunk_action_dim = -1;

            if (auto fsl_state = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(state_chunk)) {
                if (state_dim == -1) state_dim = fsl_state->list_type()->list_size();
                auto vals = std::static_pointer_cast<arrow::FloatArray>(fsl_state->values());
                all_state_vals.assign(vals->raw_values(), vals->raw_values() + vals->length());
            }
            if (auto fsl_action = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(action_chunk)) {
                if (action_dim == -1) action_dim = fsl_action->list_type()->list_size();
                auto vals = std::static_pointer_cast<arrow::FloatArray>(fsl_action->values());
                all_action_vals.assign(vals->raw_values(), vals->raw_values() + vals->length());
            }

            // Fallback: plain FloatArray (very rare)
            if (all_state_vals.empty()) {
                if (auto fa = std::dynamic_pointer_cast<arrow::FloatArray>(state_chunk)) {
                    state_dim = 1;
                    all_state_vals.reserve(fa->length());
                    for (int64_t i = 0; i < fa->length(); ++i)
                        if (!fa->IsNull(i)) all_state_vals.push_back(fa->Value(i));
                }
            }
            if (all_action_vals.empty()) {
                if (auto fa = std::dynamic_pointer_cast<arrow::FloatArray>(action_chunk)) {
                    action_dim = 1;
                    all_action_vals.reserve(fa->length());
                    for (int64_t i = 0; i < fa->length(); ++i)
                        if (!fa->IsNull(i)) all_action_vals.push_back(fa->Value(i));
                }
            }

            if (all_state_vals.empty() || all_action_vals.empty()) continue;

            int64_t rows = all_state_vals.size() / state_dim;
            for (int64_t r = 0; r < rows; ++r) {
                bool row_null = false;
                // check null bitmap for FixedSizeListArray
                if (auto fsl = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(state_chunk))
                    if (fsl->IsNull(r)) row_null = true;
                if (auto fsl = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(action_chunk))
                    if (fsl->IsNull(r)) row_null = true;
                if (row_null) continue;

                for (int d = 0; d < state_dim; ++d) {
                    float v = all_state_vals[r * state_dim + d];
                    state_sum[d]   += v;
                    state_sum_sq[d] += v * v;
                }
                for (int d = 0; d < action_dim; ++d) {
                    float v = all_action_vals[r * action_dim + d];
                    action_sum[d]   += v;
                    action_sum_sq[d] += v * v;
                }
                valid_count++;
            }
        }
    }

    if (valid_count == 0 || state_dim <= 0 || action_dim <= 0) {
        std::cerr << "No valid frames for normalization!\n";
        return;
    }

    // Trim tensors to real dimension
    state_sum   = state_sum.narrow(0, 0, state_dim);
    state_sum_sq = state_sum_sq.narrow(0, 0, state_dim);
    action_sum   = action_sum.narrow(0, 0, action_dim);
    action_sum_sq = action_sum_sq.narrow(0, 0, action_dim);

    auto state_mean = state_sum / valid_count;
    auto state_std  = (state_sum_sq / valid_count - state_mean.pow(2)).sqrt().clamp_min(1e-6f);
    auto action_mean = action_sum / valid_count;
    auto action_std  = (action_sum_sq / valid_count - action_mean.pow(2)).sqrt().clamp_min(1e-6f);

    STATE_MEAN = state_mean.to(torch::kFloat32);
    STATE_STD  = state_std.to(torch::kFloat32);
    ACTION_MEAN = action_mean.to(torch::kFloat32);
    ACTION_STD  = action_std.to(torch::kFloat32);

    std::cout << "Computed state_mean=" << STATE_MEAN << "\n";
    std::cout << "action_mean=" << ACTION_MEAN << "\n";

    // Cache to JSON
    json cache;
    cache["state_mean"]  = std::vector<float>(STATE_MEAN.data_ptr<float>(),
                                             STATE_MEAN.data_ptr<float>() + state_dim);
    cache["state_std"]   = std::vector<float>(STATE_STD.data_ptr<float>(),
                                             STATE_STD.data_ptr<float>() + state_dim);
    cache["action_mean"] = std::vector<float>(ACTION_MEAN.data_ptr<float>(),
                                             ACTION_MEAN.data_ptr<float>() + action_dim);
    cache["action_std"]  = std::vector<float>(ACTION_STD.data_ptr<float>(),
                                             ACTION_STD.data_ptr<float>() + action_dim);

    std::ofstream f(NORM_CACHE_PATH);
    f << std::setw(4) << cache << std::endl;

    std::cout << "Normalization stats computed and cached!\n";
}

void LeRobotDataset::load_normalization_stats() {
    // 1. Try to load from cache
    if (fs::exists(NORM_CACHE_PATH)) {
        try {
            std::ifstream f(NORM_CACHE_PATH);
            json cache;
            f >> cache;

            if (cache.contains("action_mean") && cache.contains("action_std") &&
                cache.contains("state_mean") && cache.contains("state_std")) {

                std::vector<float> action_mean_vec = cache["action_mean"].get<std::vector<float>>();
                std::vector<float> action_std_vec  = cache["action_std"].get<std::vector<float>>();
                std::vector<float> state_mean_vec  = cache["state_mean"].get<std::vector<float>>();
                std::vector<float> state_std_vec   = cache["state_std"].get<std::vector<float>>();

                ACTION_MEAN = torch::tensor(action_mean_vec, torch::kFloat32);
                ACTION_STD  = torch::tensor(action_std_vec,  torch::kFloat32);
                STATE_MEAN  = torch::tensor(state_mean_vec,  torch::kFloat32);
                STATE_STD   = torch::tensor(state_std_vec,   torch::kFloat32);

                std::cout << "Loaded normalization stats from cache\n";
                return;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to load cache: " << e.what() << " — recomputing...\n";
        }
    }

    // 2. Compute from scratch
    compute_normalization_from_arrow();
}

void LeRobotDataset::load_all_parquet(const fs::path& data_dir) {
    for (const auto& chunk_dir : fs::directory_iterator(data_dir)) {
        if (!chunk_dir.is_directory()) continue;
        for (const auto& file : fs::directory_iterator(chunk_dir)) {
            if (file.path().extension() != ".parquet") continue;

            // 1. Open file
            auto maybe_infile = arrow::io::ReadableFile::Open(file.path().string());
            if (!maybe_infile.ok()) {
                throw std::runtime_error("Failed to open: " + file.path().string() +
                                         " - " + maybe_infile.status().ToString());
            }
            std::shared_ptr<arrow::io::RandomAccessFile> infile = *maybe_infile;

            // 2. Open Parquet reader — NEW API (no &reader, no third arg)
            auto maybe_reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
            if (!maybe_reader.ok()) {
                throw std::runtime_error("Parquet open failed: " + maybe_reader.status().ToString());
            }
            std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*maybe_reader);

            // 3. Read table
            std::shared_ptr<arrow::Table> table;
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
    // --- Find chunk & local index ---
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

    // DYNAMIC DIMENSIONS
    int state_dim = STATE_MEAN.size(0);
    int action_dim = ACTION_MEAN.size(0);

    auto state_col = table->GetColumnByName(state_column_name_);
    auto action_col = table->GetColumnByName(action_column_name_);

    if (!state_col || !action_col) {
        std::cerr << "Missing observation.state or action column!\n";
        return f;
    }
    f.state  = read_fsl_tensor(state_col->chunk(0),  local_idx, state_dim);
    f.action = read_fsl_tensor(action_col->chunk(0), local_idx, action_dim);

    // --- Timestamp ---
    auto ts_arr = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("timestamp")->chunk(0));
    if (ts_arr && local_idx < ts_arr->length())
        f.timestamp = ts_arr->Value(local_idx);

    // --- Images ---
    if (load_images_) {  // ← ONLY LOAD IMAGES WHEN ENABLED
    std::string video_path;
    if (table->schema()->metadata()) {
        auto meta = table->schema()->metadata();
        if (meta->FindKey("video_path") != -1)
            video_path = meta->Get("video_path").ValueOr("");
    }
    if (video_path.empty() && !video_captures_.empty())
        video_path = video_captures_.begin()->first;

    for (const auto& [modality, deltas] : delta_timestamps_) {
        if (modality != "observation.image") continue;
        for (float delta : deltas) {
            double target_ts = f.timestamp + delta;
            if (target_ts < 0) continue;
            cv::Mat img = decode_frame(video_path, target_ts);
            if (!img.empty()) f.images[delta] = img;
        }
    }
}

    return f;
}

void LeRobotDataset::print_all_column_names() const {
    for (const auto& table : tables_) {
        std::cout << "Table columns: ";
        for (const auto& name : table->ColumnNames()) {
            std::cout << name << "  ";
        }
        std::cout << "\n";
        break; // just first table
    }
}
