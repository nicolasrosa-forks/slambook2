cmake_minimum_required(VERSION 2.8)
project(orb_features)

# Compilation Settings
set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Libraries
find_package(OpenCV 3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "/usr/include/eigen3/"
)

# ======== #
#  orb_cv  #
# ======== #
add_executable(orb_cv src/orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

# ========== #
#  orb_self  #
# ========== #
add_executable(orb_self src/orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

# ============= #
#  features_cv  #
# ============= #
add_executable(features_cv src/features_cv.cpp)
target_link_libraries(features_cv ${OpenCV_LIBS})

# =================== #
#  features_cv_video  #
# =================== #
add_executable(features_cv_video src/features_cv_video.cpp)
target_link_libraries(features_cv_video ${OpenCV_LIBS})