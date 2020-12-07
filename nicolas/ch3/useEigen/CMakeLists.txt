# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(useEigen)

# Compilation Settings
set(CMAKE_CXX_FLAGS "-std=c++11")  # Set C++ Standard
set(CMAKE_BUILD_TYPE "Debug")      # Set compilation mode

# Add Eigen library headers
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the target_link_libraries statement.
include_directories("/usr/include/eigen3")

# ============= #
#  eigenMatrix  #
# ============= #
# Adds an executable program
add_executable(eigenMatrix src/eigenMatrix.cpp)
