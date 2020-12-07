# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(visualizeGeometry)

# Compilation Settings
set(CMAKE_CXX_FLAGS "-std=c++11")  # Set C++ Standard
set(CMAKE_BUILD_TYPE "Debug")      # Set compilation mode

# =========== #
#  Libraries  #
# =========== #
# Eigen3
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the target_link_libraries statement.
include_directories("/usr/include/eigen3")

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# ----- Custom Libraries ----- #
# Utils
# Adds a shared library containing the functions declared in "include/libHelloSLAM.cpp"
# add_library(utils_shared SHARED include/libUtils.cpp)  # Will generate the "libhello_shared.so" file

# ============= #
#  useGeometry  #
# ============= #
# Adds an executable program
add_executable(visualizeGeometry src/visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry ${Pangolin_LIBRARIES})

