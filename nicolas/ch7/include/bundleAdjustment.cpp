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

typedef Eigen::Matrix<double, 6, 1> Vector6d;

// The memory is aligned as for dynamically aligned matrix/array types such as MatrixXd
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose){
    /* Initialization */
    const int iterations = 10;
    double cost = 0.0, lastCost = 0.0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    cout << "| ----------------------- |" << endl;
    cout << "|  Bundle Adjustment (GN) |" << endl;
    cout << "| ----------------------- |" << endl;

    /* Iteration Loop */
    print("Summary: ");
    Timer t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0.0;  // Reset

        /* Data Loop, Compute Cost */
        for (int i = 0; i < points_3d.size(); i++){
            // Describe the 3D Space Point P in the {camera2} frame
            Eigen::Vector3d Pc2 = pose * points_3d[i]; // P'_i = (T*P_i)1:3, P2_i = T21*P1_i

            double inv_z = 1.0 / Pc2[2];   // 1/Z
            double inv_z2 = inv_z * inv_z; // 1/(Z^2)

            /* ----- Compute Reprojection Error ----- */
            // Compute the Estimated Projection of P in Camera 2 (Pixel Coordinates)
            Eigen::Vector2d proj(fx * Pc2[0] / Pc2[2] + cx, fy * Pc2[1] / Pc2[2] + cy); //p2^=[u, v]^T = [fx*X/Z+cx, fy*Y/Z+cy]^T

            // Compute Residual
            Eigen::Vector2d e = points_2d[i] - proj; // e = p2_i - p2^_i

            /* ----- Jacobians ----- */
            // Compute the Jacobian Matrix
            Eigen::Matrix<double, 2, 6> J;  // 
            J << -fx * inv_z,
                0,
                fx * Pc2[0] * inv_z2,
                fx * Pc2[0] * Pc2[1] * inv_z2,
                -fx - fx * Pc2[0] * Pc2[0] * inv_z2,
                fx * Pc2[0] * Pc2[1] * inv_z,
                0,
                -fy * inv_z,
                fy * Pc2[1] * inv_z2,
                fy + fy * Pc2[1] * Pc2[1] * inv_z2,
                -fy * Pc2[0] * Pc2[1] * inv_z2,
                -fy * Pc2[0] * inv_z;

            /* ------ Hessian and Bias ----- */
            H += J.transpose() * J;  // Hessian, H(x) = J(x)'*Ω*J(x)
            b += -J.transpose() * e; // Bias, g(x) = -b(x) = -Ω*J(x)*f(x), f(x)=e(x)

            // Least-Squares Cost (Objective Function)
            cost += e.squaredNorm();  // The actual error function being minimized by solving the proposed linear system is min_x(sum_i ||ei(x)||^2).
        }

        /* ----- Solve ----- */
        // Solve the Linear System Ax=b, H(x)*∆x = g(x)
        Vector6d dx = H.ldlt().solve(b);  // δξ (Lie Algebra)

        // Check Solution
        if (isnan(dx[0])){
            cout << "Result is nan!" << endl;
            break;
        }

        /* Stopping Criteria */
        // If converged or the cost increased, the update was not good, then break.
        if (dx.norm() < 1e-6 || (iter > 0 && cost >= lastCost)){
            cout << "\ncost: " << cost << " >= lastCost: " << lastCost << ", break!" << endl;
            break;
        }

        /* ----- Update ----- */
        // Left multiply T by a disturbance quantity δξ
        pose = Sophus::SE3d::exp(dx) * pose; // T* = exp(δξ).T
        
        lastCost = cost;
        // cout << "it: " << iter << "cost: " << cost << ", \tupdate: " << dx.transpose() << "\t\testimated params: " << pose.matrix() << endl;
        cout << "it: " << iter << ",\tcost: " << cost << ",\tupdate: " << dx.transpose() << endl;
    }
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);

    cout << "\nPose (T*) by GN:\n" << pose.matrix() << endl << endl;
}

void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose){
    cout << "| ------------------------ |" << endl;
    cout << "|  Bundle Adjustment (g2o) |" << endl;
    cout << "| ------------------------ |" << endl;

}
