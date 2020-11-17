# Declares the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declares a cmake project
project(HelloSLAM)

# Set compilation mode
set(CMAKE_BUILD_TYPE "Debug")

# =========== #
#  helloSLAM  #
# =========== #
# Adds an executable program
# Syntax: add_executable (<program_name> <source_code_file>)
add_executable(helloSLAM src/helloSLAM.cpp)

# Adds a Static library containing the functions declared in "libHelloSLAM.cpp"
add_library(hello include/libHelloSLAM.cpp) # Will generate the "libhello.a" file

# Adds a Shared library containing the functions declared in "libHelloSLAM.cpp"
add_library(hello_shared SHARED include/libHelloSLAM.cpp) # Will generate the "libhello_shared.so" file

# ========== #
#  useHello  #
# ========== #
# Adds an executable program to call functions in the hello library
add_executable(useHello src/useHello.cpp)

# Links the library file to the executable program
# You could either use the Static Library ("hello") or Shared Library ("hello_shared")
# target_link_libraries(useHello hello)
target_link_libraries(useHello hello_shared)


