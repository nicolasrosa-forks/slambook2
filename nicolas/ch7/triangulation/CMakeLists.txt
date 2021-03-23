cmake_minimum_required(VERSION 2.8)
project(triangulation)

# Compilation Settings
set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Libraries
find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
add_executable(triangulation src/triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(triangulation_original src/triangulation_original.cpp)
target_link_libraries(triangulation_original ${OpenCV_LIBS})