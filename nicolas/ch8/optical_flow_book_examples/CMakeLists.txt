cmake_minimum_required(VERSION 2.8)
project(optical_flow_book_examples)

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
add_executable(optical_flow src/optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(optical_flow_original src/optical_flow_original.cpp)
target_link_libraries(optical_flow_original ${OpenCV_LIBS})

# ==================== #
#  optical_flow_video  #
# ==================== #
add_executable(optical_flow_video_opencv src/optical_flow_video_opencv.cpp)
target_link_libraries(optical_flow_video_opencv ${OpenCV_LIBS})

add_executable(optical_flow_video_multi src/optical_flow_video_multi.cpp)
target_link_libraries(optical_flow_video_multi ${OpenCV_LIBS})

# =============== #
#  direct_method  #
# =============== #
add_executable(direct_method src/direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(direct_method_original src/direct_method_original.cpp)
target_link_libraries(direct_method_original ${OpenCV_LIBS} ${Pangolin_LIBRARIES})