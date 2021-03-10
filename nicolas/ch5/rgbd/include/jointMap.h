/* System Libraries */
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>      // std::fixed, std::setprecision
#include <chrono>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Sophus Libraries */
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* OpenCV Library */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Boost */
#include <boost/format.hpp>  // For formating strings

/* Custom Libraries */
#include "../../include/libUtils.h"

/* Function Scopes */
typedef vector<double, Eigen::aligned_allocator<double>> TimeStamp;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;  // Since Eigen doesn't have Vector6d, we need to create it.
typedef vector<Vector6d, Eigen::aligned_allocator<Vector6d>> PointCloud;

TrajectoryType ReadTrajectory(TimeStamp &timestamps, const string &path);
TrajectoryType ReadTrajectory2(const string &path);
void showPointCloud(const PointCloud &pointcloud);