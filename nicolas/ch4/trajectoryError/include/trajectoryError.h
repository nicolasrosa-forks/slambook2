/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>      // std::fixed, std::setprecision

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Sophus Libraries */
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* Custom Libraries */
#include "../../include/libUtils.h"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef vector<double, Eigen::aligned_allocator<double>> TimeStamp;

TrajectoryType ReadTrajectory(TimeStamp &timestamps, const string &path);
void DrawTrajectory(const TrajectoryType &est, const TrajectoryType &gt);
