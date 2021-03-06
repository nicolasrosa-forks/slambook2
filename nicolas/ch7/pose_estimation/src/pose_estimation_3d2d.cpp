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
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"
#include "../../include/find_features_matches.h"
// #include "../../include/pose_estimation_2d2d.h"
#include "../../include/pose_estimation_3d2d.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../images/1.png";
string image2_filepath = "../../images/2.png";
string depth1_filepath = "../../images/1_depth.png";
string depth2_filepath = "../../images/2_depth.png";

int nfeatures = 500;

// Choose the PnP Method:
const char* pnp_methods_enum2str[] = {"Iterative (LM)", "EPnP", "P3P"};
int pnp_method_selected = 1;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use the Perspective-n-Point (PnP) to estimate camera motion (3D-2D Pose Estimation). */
int main(int argc, char **argv) {
    cout << "[pose_estimation_3d2d] Hello!" << endl << endl;

    /* Load the color images */
    Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);  // FIXME: I think this its not working!

    /* Load the depth images */
    // The depth image is a 16-bit unsigned number, single channel image (16UC1)
    Mat depth1 = imread(depth1_filepath, CV_LOAD_IMAGE_UNCHANGED);  
    Mat depth2 = imread(depth2_filepath, CV_LOAD_IMAGE_UNCHANGED);
    assert(depth1.data != nullptr && depth2.data != nullptr);

    // For plotting
    Mat depth1_uint8 = imread(depth1_filepath, CV_LOAD_IMAGE_GRAYSCALE);
    Mat depth2_uint8 = imread(depth2_filepath, CV_LOAD_IMAGE_GRAYSCALE);

    /* ---------------------------------- */
    /*  Features Extraction and Matching  */
    /* ---------------------------------- */
    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;

    //--- Step 0: Features Extraction, Matching and Filtering
    Timer t1 = chrono::steady_clock::now();
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, nfeatures, true);
    Timer t2 = chrono::steady_clock::now();

    /* ----------------------- */
    /*  Pose Estimation 3D-2D  */
    /* ----------------------- */
    //--- Step 1: Create 3D-2D pairs
    vector<Point3f> pts1_3d;  // {P1}_n
    vector<Point2f> pts2_2d;  // {p2}_n

    Timer t3 = chrono::steady_clock::now();
    for(DMatch m : goodMatches){  // Loop through feature matches
        // Gets the depth value of the feature point p1_i
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];  // ushort: unsigned short int, [0 to 65,535]

        // Ignores bad feature pixels
        if(d1 == 0)  // Invalid depth value
            continue;

        // Converts uint16 data to meters
        float dd1 = float(d1) / 5000.0;  // ScalingFactor from TUM Dataset.

        // Calculates the 3D Points
        Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1

        // The 3D Point P is described in {world} frame or in {camera1} frame? 
        // @nick I believe its in the {camera1} frame because the authors said its possible to compare the resulting 
        // R, t with the R, t obtained in the Pose Estimation 2D-2D (Two-View Problem), and there R, t were R21, t21!
        pts1_3d.push_back(Point3f(x1.x * dd1, x1.y * dd1, dd1));  // {P1}_n, P1 = [X, Y, Z]^T = [x*Z, y*Z, Z]^T, x = [x, y] = [X/Z, Y/Z]
        pts2_2d.push_back(keypoints2[m.trainIdx].pt);             // {p2}_n, p2 = [u2, v2]^T
    }
    Timer t4 = chrono::steady_clock::now();
    
    //--- Step 2: Perspective-n-Point (PnP) using OpenCV's solvePnP
    Mat r, R, t;  // Rotation Vector, Rotation Matrix, Translation Vector
    
    // P3P Variables
    vector<Point3f> pts1_3d_3;
    vector<Point2f> pts2_2d_3;

    // Calls OpenCV's PnP to solve, choose Iterative (LM), EPnP, P3P, DLS (broken), UPnP (broken), and other methods.
    /** @brief Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
    @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
    1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here.
    @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
    where N is the number of points. vector\<Point2d\> can be also passed here.
    @param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
    @param distCoeffs Input vector of distortion coefficients
    \f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
    assumed.
    @param rvec Output rotation vector (see @ref Rodrigues) that, together with tvec, brings points from
    the model coordinate system to the camera coordinate system.
    @param tvec Output translation vector.
    **/

    // NOTE: Since we used the normalized coordinates of the Image 1, x1, to obtain the 3D Points P, and used the feature
    // points on the Image 2. So, the 3D Points are described in the {camera1} frame, and the R, t returned by th solvePnP
    // describes the T21(R21, t21), {camera1}-to-{camera2} Transform. That's why we can compare with the R,t from the Pose
    // Estimation 2D-2D.
    switch(pnp_method_selected){
        case 1:  // Option 1: Iterative method is based on a Levenberg-Marquardt optimization
            cv::solvePnP(pts1_3d, pts2_2d, K, Mat(), r, t, false, SOLVEPNP_ITERATIVE);
            break;
        case 2:  // Option 2: EPnP
            cv::solvePnP(pts1_3d, pts2_2d, K, Mat(), r, t, false, SOLVEPNP_EPNP);  
            break;
        case 3:  //Option 3: P3P
            // P3P needs 4 3D-2D pairs: 3 pairs of 3D-2D matching points and 1 verification pair for removing solution ambiguity.
            pts1_3d_3 = slicing<vector<Point3f>>(pts1_3d, 0, 3);
            pts2_2d_3 = slicing<vector<Point2f>>(pts2_2d, 0, 3);
            // printVec("pts1_3d_3:\n", pts1_3d_3);
            // printVec("pts2_2d_3:\n", pts2_2d_3);
            cv::solvePnP(pts1_3d_3, pts2_2d_3, K, Mat(), r, t, false, SOLVEPNP_P3P);
            break;
        default:
            break;
    }

    cv::Rodrigues(r, R);  // Converts the rotation vector r to a rotation matrix R using the Rodrigues formula.
    Timer t5 = chrono::steady_clock::now();

    printMatrix("r:\n", r);  // r21?
    printMatrix("R:\n", R);  // R21?
    printMatrix("t:\n", t);  // t21?

    //--- Step 3: Bundle Adjustment (BA)
    /* In SLAM, the usual approach is to first estimate the camera pose using P3P/EPnP and then construct a least-squares
       optimization problem to adjust the estimated values (bundle adjustment). */
    VecVector3d pts1_3d_eigen;
    VecVector2d pts2_2d_eigen;

    // Copy data from OpenCV's vector to Eigen's Vector.
    for(size_t i = 0; i<pts1_3d.size(); i++){
        pts1_3d_eigen.push_back(Eigen::Vector3d(pts1_3d[i].x, pts1_3d[i].y, pts1_3d[i].z));
        pts2_2d_eigen.push_back(Eigen::Vector2d(pts2_2d[i].x, pts2_2d[i].y));
    }
    
    // printVector<Eigen::Vector3d>("pts1_3d_eigen[0]:", pts1_3d_eigen[0]);
    // printVector<Eigen::Vector2d>("pts2_2d_eigen[0]:", pts2_2d_eigen[0]);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
        K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    //--- Step 3.1: Perspective-n-Point (PnP) using Bundle Adjustment by Non-linear Optimization (Gauss Newton, GN)
    Sophus::SE3d pose_gn;
    Timer t6 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts1_3d_eigen, pts2_2d_eigen, K_eigen, pose_gn);
    Timer t7 = chrono::steady_clock::now();
    
    //--- Step 3.2: Perspective-n-Point (PnP) using Bundle Adjustment by Graph Optimization (g2o)
    Sophus::SE3d pose_g2o;
    Timer t8 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts1_3d_eigen, pts2_2d_eigen, K_eigen, pose_g2o);
    Timer t9 = chrono::steady_clock::now();
    
    /* --------- */
    /*  Results  */
    /* --------  */
    cout << "-------------------------------------------------" << endl;
    printElapsedTime("Pose estimation 3D-2d: ", t1, t9);
    printElapsedTime(" | Features Calculation: ", t1, t2);
    printElapsedTime(" | Create 3D-2D Pairs: ", t3, t4);
    printElapsedTime(" | Perspective-n-Point (solvePnP): ", t4, t5);
    printElapsedTime(" | Bundle Adjustment (GN): ", t6, t7);
    printElapsedTime(" | Bundle Adjustment (g2o): ", t8, t9);
    
    // NOTE: Observe that not all the 79 feature matches have valid depth values. 4 3D-2D pairs were discarded.
    cout << "\n-- Number of 3D-2D pairs: " << pts1_3d.size() << endl;
    cout << "-- PnP Method selected: " << pnp_methods_enum2str[pnp_method_selected-1] << endl;
    cout << "-------------------------------------------------" << endl;

    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    imshow("depth1", depth1_uint8);
    // imshow("depth2", depth2_uint8);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}

/* "Readers can compare R, t solved in the previous 2D-2D case to see the difference.
It can be seen that when 3D information is involved, the estimated R is almost the
same, while the t is quite different." */

/* ==================================== */
/*  Results from Pose Estimation 2D-2D  */
/* ==================================== */
// R:
// [0.9969387384754708, -0.05155574188737422, 0.05878058527591362;
//  0.05000441581290405, 0.998368531736214, 0.02756507279306545;
//  -0.06010582439453526, -0.02454140006844053, 0.9978902793175882]
// (3, 3)

// t:
// [-0.9350802885437915;
//  -0.03514646275858852;
//  0.3526890700495534]
// (3, 1)

/* ==================================== */
/*  Results from Pose Estimation 3D-2D  */
/* ==================================== */
// This following results should be more accurate than the previous one, since utilized the depth information (3D-2D).

/* ----- Only PnP ----- */
// Step 2: Pose (T*) by PnP:
// R:
// [0.9979059095501289, -0.05091940089111061, 0.03988747043647122;
//  0.04981866254254162, 0.9983623157438141, 0.02812094175381183;
//  -0.04125404886071624, -0.02607491352889362, 0.9988083912027663]
// (3, 3)

// t:
// [-0.1267821389556797;
//  -0.008439496817594663;
//  0.06034935748886035]
// (3, 1)

/* ----- Bundle Adjustment ("PnP" + Non-linear/Graph Optimization)
// Step 3.1: Pose (T*) by GN:
//    0.997906  -0.0509194   0.0398875   -0.126782
//   0.0498187    0.998362    0.028121 -0.00843953
//  -0.0412541  -0.0260749    0.998808   0.0603494
//           0           0           0           1

// Step 3.2: Pose (T*) by g2o:
//     0.99790590955  -0.0509194008911   0.0398874704367   -0.126782138956
//   0.0498186625425    0.998362315744   0.0281209417542 -0.00843949681823
//  -0.0412540488609  -0.0260749135293    0.998808391203   0.0603493574888
//                 0                 0                 0                 1

/* ==================================== */ 
/*  Original Code (pixel2cam, Point2f)  */
/* ==================================== */ 
// PnP:
// R=
// [0.9979059095501289, -0.05091940089111061, 0.03988747043647122;
//  0.04981866254254162, 0.9983623157438141, 0.02812094175381183;
//  -0.04125404886071624, -0.02607491352889362, 0.9988083912027663]
// t=
// [-0.1267821389556797;
//  -0.008439496817594663;
//  0.06034935748886035]

// pose by g-n: 
//    0.997905909549  -0.0509194008562   0.0398874705187   -0.126782139096
//    0.049818662505    0.998362315745   0.0281209417649 -0.00843949683874
//  -0.0412540489424  -0.0260749135374    0.998808391199   0.0603493575229
//                 0                 0                 0                 1

// pose estimated by g2o =
//     0.99790590955  -0.0509194008911   0.0398874704367   -0.126782138956
//   0.0498186625425    0.998362315744   0.0281209417542 -0.00843949681823
//  -0.0412540488609  -0.0260749135293    0.998808391203   0.0603493574888
//                 0                 0                 0                 1

/* ==================================== */ 
/*  Original Code (pixel2cam, Point2d)  */
/* ==================================== */ 
// PnP:
// R=
// [0.9979059096319058, -0.05091940167648939, 0.03988746738797636;
//  0.04981866392256838, 0.9983623160259321, 0.02812092929309315;
//  -0.04125404521606011, -0.02607490119339458, 0.9988083916753333]
// t=
// [-0.1267821338701787;
//  -0.008439477707051628;
//  0.0603493450570466]

// pose by g-n: 
//     0.99790590963  -0.0509194016416   0.0398874674701    -0.12678213401
//    0.049818663885    0.998362316027   0.0281209293043 -0.00843947772828
//  -0.0412540452977  -0.0260749012019    0.998808391672   0.0603493450911
//                 0                 0                 0                 1

// pose estimated by g2o =
//    0.997905909632  -0.0509194016765   0.0398874673881    -0.12678213387
//   0.0498186639226    0.998362316026   0.0281209292935 -0.00843947770777
//  -0.0412540452162  -0.0260749011938    0.998808391675    0.060349345057
//                 0                 0                 0                 1
