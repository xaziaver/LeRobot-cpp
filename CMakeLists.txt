cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LeRobotCpp)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(lerobot src/main.cpp src/dataset.h src/dataset.cpp)
target_link_libraries(lerobot PRIVATE
    "${TORCH_LIBRARIES}"
    opencv_core opencv_videoio
    Arrow::arrow_shared
    Parquet::parquet_shared
    nlohmann_json::nlohmann_json
)
set_property(TARGET lerobot PROPERTY CXX_STANDARD 17)
