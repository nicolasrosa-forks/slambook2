# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(ex5)

# Compilation Settings
set(CMAKE_CXX_FLAGS "-std=c++11")  # Set C++ Standard
set(CMAKE_BUILD_TYPE "Debug")      # Set compilation mode

# =========== #
#  Libraries  #
# =========== #
# Eigen3
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the target_link_libraries statement.
include_directories("/usr/include/eigen3")

# ===== #
#  ex5  #
# ===== #
# Adds an executable program
add_executable(ex5 src/ex5.cpp)
