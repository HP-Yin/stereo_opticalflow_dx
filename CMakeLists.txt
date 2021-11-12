cmake_minimum_required(VERSION 2.8)
project(fts_extract_match)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
)

add_library(stereo_lk_dx SHARED
    stereo_lk/OpticalFlowPyrLKDx.cpp
)

add_executable(stereo_optical_flowback stereo_optical_flowback.cpp)
target_link_libraries(stereo_optical_flowback ${OpenCV_LIBS} stereo_lk_dx)

