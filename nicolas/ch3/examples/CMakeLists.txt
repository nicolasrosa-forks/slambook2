# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(examples)

# Compilation Settings
set(CMAKE_CXX_FLAGS "-std=c++11")  # Set C++ Standard
set(CMAKE_BUILD_TYPE "Release")    # Set compilation mode

# =========== #
#  Libraries  #
# =========== #
# We need to use find_package to make cmake find Pangolin

# Eigen3
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the 'target_link_libraries' statement.
include_directories("/usr/include/eigen3")

# Pangolin 
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# ===================== #
#  coordinateTransform  #
# ===================== #
# Adds an executable program
add_executable(coordinateTransform src/coordinateTransform.cpp)

# ================ #
#  plotTrajectory  #
# ================ #
add_executable(plotTrajectory src/plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
