/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <unistd.h>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* OpenCV Library */
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_opencv.h"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
typedef vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> PointCloud;
void showPointCloud(const PointCloud &pointcloud);
