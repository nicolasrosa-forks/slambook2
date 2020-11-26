# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(useGeometry)

# Compilation Settings
set(CMAKE_BUILD_TYPE "Debug")

# Add Eigen library headers
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the target_link_libraries statement.
include_directories("/usr/include/eigen3")

# ============= #
#  useGeometry  #
# ============= #
# Adds an executable program
add_executable(useGeometry useGeometry.cpp)
