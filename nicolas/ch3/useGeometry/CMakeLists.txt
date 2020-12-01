# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(useGeometry)

# Compilation Settings
set(CMAKE_BUILD_TYPE "Debug")  # Set compilation mode

# Add Eigen library headers
# Because the Eigen library only has header files, so you don’t need to link the program to the library with the target_link_libraries statement.
include_directories("/usr/include/eigen3")

# ======= #
#  Utils  #
# ======= #
# Adds a shared library containing the functions declared in "include/libHelloSLAM.cpp"
# add_library(utils_shared SHARED include/libUtils.cpp)  # Will generate the "libhello_shared.so" file

# ============= #
#  useGeometry  #
# ============= #
# Adds an executable program
add_executable(useGeometry src/useGeometry.cpp)
# target_link_libraries(useGeometry utils_shared)
# target_link_libraries(useGeometry ${CMAKE_SOURCE_DIR}/libUtils_shared.so)
