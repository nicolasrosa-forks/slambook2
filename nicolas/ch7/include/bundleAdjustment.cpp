/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/* g2o Libraries */
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

/* Sophus Libraries */
#include <sophus/se3.hpp>

/* Custom Libraries */
#include "../../common/libUtils.h"

using namespace std;
using namespace cv;

// BA by g2o
// The memory is aligned as for dynamically aligned matrix/array types such as MatrixXd
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose){
    typedef Eigen::Matrix<double, 6, 1> Vector6d;  // FIXME: mover?
    
    /* Initialization */
    const int iterations = 10;
    double cost = 0.0, lastCost = 0.0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    /* Loop */
    for(int iter=0; iter<iterations; iter++){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0.0;

        // Compute Cost
        for (int i=0; i<points_3d.size(); i++){
            // Describe the 3D Space Point P in the {camera2} frame
            Eigen::Vector3d pc = pose*points_3d[i]; // P'_i = (T*P_i)1:3, P2_i = T21*P1_i
            
            double inv_z = 1.0/pc[2];               // 1/Z
            double inv_z2 = inv_z*inv_z;            // 1/(Z^2)

            // Compute the Estimated Projection of P in Camera 2
            Eigen::Vector2d proj(fx*pc[0]/pc[2]+cx, fy*pc[1]/pc[2]+cy); //p2^_=[u, v]^T = [fx*X/Z+cx, fy*Y/Z+cy]^T

            // Compute the Reprojection Error
            Eigen::Vector2d e = points_2d[i] - proj;  // Residual, e = p2_i - p2^_i

            cost += e.squaredNorm();

            // Compute the Jacobian Matrix
            Eigen::Matrix<double, 2, 6> J;
            J <<
                -fx*inv_z,
                0,
                fx*pc[0]*inv_z2,
                fx*pc[0]*pc[1]*inv_z2,
                -fx-fx*pc[0]*pc[0]*inv_z2,
                fx*pc[0]*pc[1]*inv_z,
                0,
                -fy*inv_z,
                fy*pc[1]*inv_z2,
                fy+fy*pc[1]*pc[1]*inv_z2,
                -fy*pc[0]*pc[1]*inv_z2,
                -fy*pc[0]*inv_z;
            
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if(isnan(dx[0])){
            cout << "result is nan!" << endl;
            break;
        }

        if(iter > 0 && cost >=lastCost){
            // cost increase, update is not good
            cout << "cost: " << cost << ", lastCost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx)*pose;  // FIXME: Left Multiplication perturbance
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if(dx.norm() < 1e-6){
            //converge
            break;
        }
    }

    cout << "pose_gn:\n" << pose.matrix() << endl;
}

void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose){ 
    print("BA_G2O");
}
