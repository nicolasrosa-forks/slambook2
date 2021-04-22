/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
//#include <dirent.h>
#include <string>
//#include <system_error>

/* Eigen Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/* OpenCV Libraries */
//#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"
#include "../include/OpticalFlowTracker.h"

using namespace std;
using namespace cv;

/** Description
 * @brief Get a grayscale value from reference image (bilinear interpolation)
 * 
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y){
    // Boundary check
    if (x < 0) x = 0;                    // Avoid negative x-axis coordinates
    if (y < 0) y = 0;                    // Avoid negative y-axis coordinates
    if (x >= img.cols) x = img.cols -1;  // Avoid positive x-axis coordinates outside image width
    if (y >= img.rows) y = img.rows -1;  // Avoid positive y-axis coordinates outside image height

    uchar *data = &img.data[int(y)*img.step + int(x)];
    
    float xx = x - floor(x);
    float yy = y - floor(y);

    return float(
        (1 - xx) * (1 - yy) * data[0] + 
        xx * (1 - yy) * data[1] + 
        (1 - xx) * yy * data[img.step] + 
        xx * yy * data[img.step + 1]
    );
}

/* ============================== */
/*  Class Methods Implementation  */
/* ============================== */
void OpticalFlowTracker::calculateOpticalFlow(const Range &range){
    /* Initialization */
    int half_patch_size = 4;
    int iterations = 10;

    /* Iterate over Features */
    for(size_t i=range.start; i < range.end; i++){
        auto kp = kps1[i];
        double dx = 0, dy = 0;  // dx, dy need to be estimated from

        if(has_initial_guess){
            dx = kps2[i].pt.x - kp.pt.x;
            dy = kps2[i].pt.y - kp.pt.y;
        }

        double cost = 0.0, lastCost = 0.0;
        bool succ = true;  // Indicate if current point succeeded

        /* Gauss-Newton method Initialization */
        // Used to solve the Optical Flow Optimization problem (Minimizes the photometric error).
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();  // Hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();  // bias

        Eigen::Vector2d J;  // Jacobian

        /* Iterations */
        for (int iter = 0; iter < iterations; iter++){
            if(inverse == false){
                // Resets H and b
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }else{
                /** Inverse Optical Flow
                 * @brief In this method, the gradient of I1(x,y) remains unchanged, so we can use the result calculated
                 * in the first iteration in the subsequent iterations. When the Jacobian remains unchanged, the H matrix
                 * is unchanged, and only the residual is calculated for each iteration, which can save a lot of calculation.
                 */                
                // So, we only reset b.
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            /* ----- Compute Photometric Error ----- */
            // Iterate over Patch
            for(int x = -half_patch_size; x <= half_patch_size; x++)       // FIXME: x < half_patch_size OR x <= half_patch_size?
                for(int y = -half_patch_size; y <= half_patch_size; y++){  // FIXME: y < half_patch_size OR y <= half_patch_size?
                    /* 1. Compute Residual */
                    // Residual Error, e = I1(x, y) - I2(x + ∆x, y + ∆y)
                    double error =  GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                    GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    /* 2. Compute the Least-Squares Cost */
                    cost += error * error;

                    /* 3. Compute Jacobian */ // TODO: How it's calculated?
                    if(inverse == false){
                        // In this mode, we need to calculate the Jacobian every iteration.
                        J = -1.0 * Eigen::Vector2d(
                             0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                    GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                             0.5 * (GetPixelValue(img2, kp.pt.x + dx + x,     kp.pt.y + dy + y + 1) -
                                    GetPixelValue(img2, kp.pt.x + dx + x,     kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0){
                        // In inverse mode, J it's the same for all iterations
                        // NOTE: This J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                             0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                    GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                             0.5 * (GetPixelValue(img1, kp.pt.x + x,     kp.pt.y + y + 1) -
                                    GetPixelValue(img1, kp.pt.x + x,     kp.pt.y + y - 1))
                        );
                    }

                    /* 4. Compute Hessian and Bias */
                    // Information Matrix(Ω) wasn't informed, so consider it as identity.
                    if(inverse == false || iter == 0){
                        // also update the Hessian
                        H += J * J.transpose();  // Hessian, H(x) = J(x)'*Ω*J(x)
                    }
                    b += -J * error;  // Bias, g(x) = -b(x) = -e(x)'*Ω*J(x) = -(Ω*J(x))'*e(x) = -Ω*J(x)'*e(x)
                    // printMatrix<Vector2d>("J: ", J);
                }

            /* ----- Solve! ----- */
            // Solve the Linear System A*x=b, H(x)*∆x = g(x)
            Eigen::Vector2d update = H.ldlt().solve(b);  // ∆x=[dx, dy]

            // Check Solution
            if (isnan(update[0])){
                // Sometimes occurs when we have a black or white patch, or when H is irresversible
                // cout << "Update is nan!" << endl;
                succ = false;
                break;
            }

            /* Stopping Criteria */
            // If the cost increased, the update was not good, then break.
            if (iter > 0 && cost >= lastCost){  //FIXME: cost > lastCost? or cost >= lastCost?
                // cout << "\ncost: " << cost << " >= lastCost: " << lastCost << ", break!" << endl;
                break;
            }

            /* ----- Update ----- */
            dx += update[0];  // dx
            dy += update[1];  // dy

            lastCost = cost;
            succ = true;

            // cout << "it: " << iter << ",\tcost: " << std::setprecision(12) << cost << ",\tupdate: " << update.transpose() << endl;  // TODO

            if (update.norm() < 1e-2){  // Method converged!
                break;
            }
        }

        success[i] = succ;

        // Set kps2
        kps2[i].pt = kp.pt + Point2f(dx, dy);  // I2(x+dx, y + dy)

        // cout << endl;
    }
}