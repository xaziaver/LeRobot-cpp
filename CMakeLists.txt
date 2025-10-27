cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LeRobot-cpp)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Arrow REQUIRED)
find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(lerobot main.cpp src/dataset.cpp src/policy.cpp)
target_link_libraries(lerobot "${TORCH_LIBRARIES}" opencv_core opencv_videoio arrow_shared nlohmann_json::nlohmann_json)

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
