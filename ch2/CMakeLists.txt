# Declare the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declare a cmake project
project(HelloSLAM)

# Set compilation mode
set(CMAKE_BUILD_TYPE "Debug")

# Add an executable program
# Syntax: add_executable(program name source code file)
add_executable(helloSLAM helloSLAM.cpp)

# Add hello library
add_library(hello libHelloSLAM.cpp)
# Shared library
add_library(hello_shared SHARED libHelloSLAM.cpp)

# Add an executable program to call functions in the hello library
add_executable(useHello useHello.cpp)
# Link the library file to the executable program
target_link_libraries(useHello hello_shared)
