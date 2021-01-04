/* System Libraries */
#include <iostream>
#include <fstream>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Sophys Libraries */
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
string estimated_file = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch4/trajectoryError/src/estimated.txt";
string groundtruth_file = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch4/trajectoryError/src/groundtruth.txt";

/* Function Scopes */
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

TrajectoryType ReadTrajectory(const string &path);

/* ====== */
/*  Main  */
/* ====== */
/*  This Program demonstrates the calculation of the Absolute Trajectory Error (ATE) */
int main(int argc, char **argv){
    print("helloTrajectoryError!");

    // 1. Read the two trajectories (Sequences of Poses)
    TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
    TrajectoryType estimated = ReadTrajectory(estimated_file);

    assert(!groundtruth.empty() && !estimated.empty());
    assert(groundtruth.size() == estimated.size());

    // 2. Calculate the Absolute Trajectory Error (ATE)
    double rmse = 0;
    Sophus::SE3d pose_est, pose_gt;
    int N = estimated.size();
    
    for(size_t i=0; i < N; i++){
        pose_gt = groundtruth[i];
        pose_est = estimated[i];

        string pose_gt_str = ("pose_gt[" + to_string(i) + "]: ");
        string pose_est_str = ("pose_est[" + to_string(i) + "]: ");

        printMatrix<Matrix4d>(pose_gt_str.c_str(), pose_gt.matrix());  
        printMatrix<Matrix4d>(pose_est_str.c_str(), pose_est.matrix());

        double error = (pose_gt.inverse()*pose_est).log().norm();
        rmse += error * error;
    }

    rmse = sqrt(rmse/double(N));

    cout << "RMSE: " << rmse << endl;

    // 3. Display the trajectories in a 3D Window.

}

/* =========== */
/*  Functions  */
/* =========== */
TrajectoryType ReadTrajectory(const string &path){
    ifstream fin(path);
    TrajectoryType trajectory;

    if(!fin){
        cout << "Cannot find trajectory file at '" << path << "'." << endl;
        return trajectory;
    }else{
        cout << "Read '" << path << "' was sucessful." << endl;
    }

    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Pose, Transformation Matrix (T)
        Sophus::SE3d pose(Eigen::Quaterniond(qx, qy, qz, qw), Eigen::Vector3d(tx, ty, tz)); // T, SE(3) from q,t.
        trajectory.push_back(pose);
    }
    cout << "Read total of " << trajectory.size() << " pose entries." << endl << endl;

    return trajectory;
}