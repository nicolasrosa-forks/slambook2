cmake_minimum_required(VERSION 2.8)
project(pose_estimation)

# Compilation Settings
set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Libraries
find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

# ====================== #
#  pose_estimation_2d2d  #
# ====================== #
# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d_tum src/pose_estimation_2d2d_tum.cpp)
target_link_libraries(pose_estimation_2d2d_tum ${OpenCV_LIBS})

# ====================== #
#  pose_estimation_3d2d  #
# ====================== #
add_executable(pose_estimation_3d2d src/pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d g2o_core g2o_stuff ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d_original src/pose_estimation_3d2d_original.cpp)
target_link_libraries(pose_estimation_3d2d_original g2o_core g2o_stuff ${OpenCV_LIBS})

# ====================== #
#  pose_estimation_3d3d  #
# ====================== #
add_executable(pose_estimation_3d3d src/pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d g2o_core g2o_stuff ${OpenCV_LIBS})

add_executable(pose_estimation_3d3d_original src/pose_estimation_3d3d_original.cpp)
target_link_libraries(pose_estimation_3d3d_original g2o_core g2o_stuff ${OpenCV_LIBS})
