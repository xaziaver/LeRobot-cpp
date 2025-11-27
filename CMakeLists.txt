cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LeRobotCpp LANGUAGES CXX)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Torch REQUIRED)

# C++17 + optimizations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Executable
add_executable(train
    src/train.cpp
    src/dataset.cpp
    src/act_policy.cpp
)

target_include_directories(train PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(train PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Arrow::arrow_shared
    Parquet::parquet_shared
    nlohmann_json::nlohmann_json
)

set_target_properties(train PROPERTIES
    INSTALL_RPATH "$ORIGIN/../libtorch/lib"
    BUILD_WITH_INSTALL_RPATH ON
)
