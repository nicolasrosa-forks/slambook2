cmake_minimum_required(VERSION 2.8)
project(optical_flow_official_examples)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")


find_package(OpenCV 3 REQUIRED)
# find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
    ${Pangolin_INCLUDE_DIRS}
)
# ============== #
#  optical_flow  #
# ============== #
add_executable(cv_optical_flow src/cv_optical_flow.cpp)
target_link_libraries(cv_optical_flow ${OpenCV_LIBS})

# ==================== #
#  optical_flow_dense  #
# ==================== #
add_executable(cv_optical_flow_dense src/cv_optical_flow_dense.cpp)
target_link_libraries(cv_optical_flow_dense ${OpenCV_LIBS})