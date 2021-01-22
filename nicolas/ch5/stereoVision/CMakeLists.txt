# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(imageBasics)

# Compilation Settings
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")  # Set C++ Standard
set(CMAKE_BUILD_TYPE "Release")        # Set compilation mode

# =========== #
#  Libraries  #
# =========== #
# We need to use find_package to make cmake find Pangolin and Sophus

# Eigen3
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the 'target_link_libraries' statement.
include_directories("/usr/include/eigen3")

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})  # Sophus is header only

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})  # OpenCV is header only

# ----- Custom Libraries ----- #
# Utils
# Adds a shared library containing the functions declared in "include/libHelloSLAM.cpp"
# add_library(utils_shared SHARED include/libUtils.cpp)  # Will generate the "libhello_shared.so" file

# ============== #
#  stereoVision  #
# ============== #
# Adds an executable program
add_executable(stereoVision src/stereoVision.cpp)
target_link_libraries(stereoVision ${Pangolin_LIBRARIES})
target_link_libraries(stereoVision ${OpenCV_LIBS})