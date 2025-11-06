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
LeRobotDataset::LeRobotDataset(const std::string& root_path) {
  // -----------------------------------------------------------------
  // 1. Build the path to the parquet file (the only one we need now)
  // -----------------------------------------------------------------
  parquet_path_ = fs::path(root_path) / "data" / "chunk-000" / "file-000.parquet";
  if (!fs::exists(parquet_path_))
    throw std::runtime_error("Parquet file not found: " + parquet_path_.string());

  // -----------------------------------------------------------------
  // 2. Open the file with Arrow
  // -----------------------------------------------------------------
  std::shared_ptr<arrow::io::ReadableFile> infile;
  auto file_result = arrow::io::ReadableFile::Open(parquet_path_.string(),
						   arrow::default_memory_pool());
  if (!file_result.ok()) {
    throw std::runtime_error(
      "Failed to open parquet file: " +
      file_result.status().ToString());
  }
  infile = *file_result;
  
  // -----------------------------------------------------------------
  // 3. Create a Parquet → Arrow reader
  // -----------------------------------------------------------------
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto reader_status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
  if (!reader_status.ok()) {
    throw std::runtime_error(
      "Failed to create parquet reader: " +
      reader_status.status().ToString());
  }
  reader = std::move(*reader_status);
  
  // -----------------------------------------------------------------
  // 4. Read the *entire* file into one Arrow Table
  // -----------------------------------------------------------------
  PARQUET_THROW_NOT_OK(reader->ReadTable(&table_));
  std::cout << "Loaded table: " << table_->num_rows() << " rows, "
            << table_->num_columns() << " cols\n";
  
  // -----------------------------------------------------------------
  // 5. (Optional) load meta/info.json – useful for fps, shapes, etc.
  // -----------------------------------------------------------------
  fs::path meta_path = fs::path(root_path) / "meta" / "info.json";
  if (fs::exists(meta_path)) {
      std::ifstream f(meta_path);
      if (!f) {
	throw std::runtime_error("Failed to open meta file: " + meta_path.string());
      }
      f >> meta_;
  }
}

/* --------------------------------------------------------------
 *  get() – one row → Frame
 * -------------------------------------------------------------- */
Frame LeRobotDataset::get(size_t index) {
    if (index >= static_cast<size_t>(table_->num_rows()))
        throw std::out_of_range("index out of range");

    Frame f;
    // dummy image
    f.image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 255, 0));

    // ----- state ----------------------------------------------------
    auto state_col = table_->GetColumnByName("observation.state");
    if (state_col && state_col->num_chunks() > 0) {
      f.state = dense_float_to_tensor(state_col->chunk(0), static_cast<int64_t>(index));
      std::cout << "DEBUG: state tensor shape " << f.state.sizes() << "\n";
    } else {
      std::cerr << "WARNING: No observation.state column!\n";
    }

    // ----- action ---------------------------------------------------
    auto action_col = table_->GetColumnByName("action");
    if (action_col && action_col->num_chunks() > 0) {
      f.action = dense_float_to_tensor(action_col->chunk(0), static_cast<int64_t>(index));
      std::cout << "DEBUG: action tensor shape " << f.action.sizes() << "\n";
    } else {
      std::cerr << "WARNING: No action column!\n";
    }

    return f;

    // ----- dummy image (replace later with video decoding) ----------
    f.image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 255, 0));

    return f;
}

/* --------------------------------------------------------------
 *  print_column_names()
 * -------------------------------------------------------------- */
void LeRobotDataset::print_column_names() const {
    for (int i = 0; i < table_->num_columns(); ++i) {
        std::cout << table_->ColumnNames()[i] << "\t";
    }
    std::cout << "\n";
}

/* --------------------------------------------------------------
 *  tiny demo
 * -------------------------------------------------------------- */
int main() {
    try {
        LeRobotDataset ds("/home/xaziaver/Code/LeRobot-cpp/data/pusht");

        ds.print_column_names();

        const size_t N = std::min<size_t>(5, ds.size().value_or(0));
        for (size_t i = 0; i < N; ++i) {
            Frame f = ds.get(i);

            std::cout << "Row " << i << ":\n";
            std::cout << "  state  = " << f.state << "\n";
            std::cout << "  action = " << f.action << "\n";

            std::string img_path = "frame_" + std::to_string(i) + ".png";
            cv::imwrite(img_path, f.image);
            std::cout << "  image → " << img_path << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
