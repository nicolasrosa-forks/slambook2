cmake_minimum_required(VERSION 2.8)
project(optimization)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Compilation Settings
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")  # Set C++ Standard
# set(CMAKE_CXX_FLAGS "-std=c++11 -O2")  # Set C++ Standard
set(CMAKE_BUILD_TYPE "Release")

# Eigen3
include_directories("/usr/include/eigen3")

# OpenCV
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# ============= #
#  gaussNewton  #
# ============= #
add_executable(gaussNewton src/gaussNewton.cpp)
target_link_libraries(gaussNewton)

# =================== #
#  ceresCurveFitting  #
# =================== #
add_executable(ceresCurveFitting src/ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${CERES_LIBRARIES})

# ================= #
#  g2oCurveFitting  #
# ================= #
add_executable(g2oCurveFitting src/g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
