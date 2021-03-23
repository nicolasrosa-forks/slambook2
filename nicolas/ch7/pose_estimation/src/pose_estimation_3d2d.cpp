/* =========== */
/*  Libraries  */
/* =========== */
#define OPENCV3  // If not defined, OpenCV2

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
#include "../../../common/libUtils.h"
#include "../../include/find_features_matches.h"
// #include "../../include/pose_estimation_2d2d.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../orb_features/src/1.png";
string image2_filepath = "../../orb_features/src/2.png";
string depth1_filepath = "../../orb_features/src/1_depth.png";
string depth2_filepath = "../../orb_features/src/2_depth.png";

int orb_nfeatures = 500;

// Choose the PnP Method:
// 1: Iterative (LM), 2: EPnP, 3: P3P
const char* pnp_methods_enum2str[] = {"Iterative", "EPnP", "P3P"};
int pnp_method_selected = 3;

// BA by g2o
// The memory is aligned as for dynamically aligned matrix/array types such as MatrixXd
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
template <typename TTypeVec>
TTypeVec slicing(TTypeVec &arr, int begin_idx, int end_idx){
    // Starting and Ending iterators
    auto start = arr.begin() + begin_idx;
    auto end = arr.begin() + end_idx + 1;

    // To store the sliced vector
    TTypeVec result(end_idx - begin_idx + 1);
  
    // Copy vector using copy function()
    copy(start, end, result.begin());
  
    // Return the final sliced vector
    return result;
}


/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use 2D-2D feature matching to estimate camera motion. */
int main(int argc, char **argv) {
    cout << "[pose_estimation_3d2d] Hello!" << endl;

    /* Load the color images */
    Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    /* Load the depth images */
    // The depth image is a 16-bit unsigned number, single channel image (16UC1)
    Mat depth1 = imread(depth1_filepath, CV_LOAD_IMAGE_UNCHANGED);  
    Mat depth2 = imread(depth2_filepath, CV_LOAD_IMAGE_UNCHANGED);
    assert(depth1.data != nullptr && depth2.data != nullptr);

    // For plotting
    Mat depth1_uint8 = imread(depth1_filepath, CV_LOAD_IMAGE_GRAYSCALE);  
    Mat depth2_uint8 = imread(depth2_filepath, CV_LOAD_IMAGE_GRAYSCALE);

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;

    /* ---------------------------------- */
    /*  Features Extraction and Matching  */
    /* ---------------------------------- */
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, orb_nfeatures, true);
    cout << "\nIn total, we get " << goodMatches.size() << " pairs of feature points." << endl << endl;

    /* ----------------------- */
    /*  Pose Estimation 3D-2D  */
    /* ----------------------- */
    /* Create 3D-2D pairs */
    Timer t1 = chrono::steady_clock::now();
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for(DMatch m : goodMatches){  // Loop through feature matches
        // Gets the depth value of the feature point p1_i
        ushort d = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];  // ushort: unsigned short int, [0 to 65,535]

        // Ignores bad feature pixels
        if(d == 0)  // Invalid depth value
            continue;

        // Converts uint16 data to meters
        float dd = d / 5000.0;  // ScalingFactor from TUM Dataset.

        // Calculates the 3D Points
        // x = [X/Z, Y/Z]
        Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1

        pts_3d.push_back(Point3f(x1.x * dd, x1.y * dd, dd));  // P = [X, Y, Z]^T // FIXME: In {world} frame or in {camera} frame?
        pts_2d.push_back(keypoints2[m.trainIdx].pt);          // (p2)_n
    }
    
    /* Perspective-n-Point (PnP) */
    Timer t2 = chrono::steady_clock::now();
    Mat r, R, t;  // Rotation Vector, Rotation Matrix, Translation Vector
    
    // Calls OpenCV's PnP to solve, choose Iterative (LM), EPnP, P3P, DLS (broken), UPnP (broken), and other methods.
    vector<Point3f> pts_3d_3;
    vector<Point2f> pts_2d_3;

    switch(pnp_method_selected){
        case 1:  // Option 1: Iterative method is based on a Levenberg-Marquardt optimization
            cv::solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, SOLVEPNP_ITERATIVE);
            break;
        case 2:  // Option 2: EPnP
            cv::solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, SOLVEPNP_EPNP);  

            break;
        case 3:  //Option 3: P3P
            // P3P needs 4 3D-2D pairs: 3 pairs of 3D-2D matching points and 1 verification pair for removing solution ambiguity.
            pts_3d_3 = slicing<vector<Point3f>>(pts_3d, 0, 3);
            pts_2d_3 = slicing<vector<Point2f>>(pts_2d, 0, 3);
            // printVec("pts_3d_3:\n", pts_3d_3);
            // printVec("pts_2d_3:\n", pts_2d_3);
            cv::solvePnP(pts_3d_3, pts_2d_3, K, Mat(), r, t, false, SOLVEPNP_P3P);  //
            break;
        default:
            break;
    }

    cv::Rodrigues(r, R);  // Converts the rotation vector r to a rotation matrix R using the Rodrigues formula.
    Timer t3 = chrono::steady_clock::now();

    /* Bundle Adjustment */
    // In SLAM, the usual approach is to first estimate the camera pose using P3P/EPnP and then construct a least-squares
    // optimization problem to adjust the estimated values (bundle adjustment).
    
    Timer t4 = chrono::steady_clock::now();

    /* --------- */
    /*  Results  */
    /* --------  */
    printElapsedTime("Pose estimation 3D-2d: ", t1, t3);
    printElapsedTime(" | Create 3D-2D Pairs: ", t1, t2);
    printElapsedTime(" | Perspective-n-Point (solvePnP): ", t2, t3);
    printElapsedTime(" | Bundle Adjustment: ", t3, t4);
    // NOTE: Observe that not all the 79 feature matches have valid depth values. 4 3D-2D pairs were discarded.
    cout << "\n-- Number of 3D-2D pairs: " << pts_3d.size() << endl;
    cout << "-- PnP Method selected: " << pnp_methods_enum2str[pnp_method_selected-1] << endl;
    cout << endl;

    printMatrix("r:\n", r);  // FIXME: rcw?
    printMatrix("R:\n", R);  // FIXME: Rcw?
    printMatrix("t:\n", t);  // FIXME: Tcw?

    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    imshow("depth1", depth1_uint8);
    imshow("depth2", depth2_uint8);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}

/* ======================= */
/*  Functions Declaration  */
/* ======================= */