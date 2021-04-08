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
// #include "../../include/pose_estimation_3d2d.h"
#include "../../include/pose_estimation_3d3d.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../orb_features/src/1.png";
string image2_filepath = "../../orb_features/src/2.png";
string depth1_filepath = "../../orb_features/src/1_depth.png";
string depth2_filepath = "../../orb_features/src/2_depth.png";

int nfeatures = 500;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use the Iterative Closest Point (ICP) to estimate camera motion (3D-3D Pose Estimation). */
int main(int argc, char **argv) {
    cout << "[pose_estimation_3d3d] Hello!" << endl << endl;

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
    /*  Pose Estimation 3D-3D  */
    /* ----------------------- */
    //--- Step 1: Create 3D-3D pairs
    vector<Point3f> pts1_3d;  // {P1}_n, P' = {p1', p2', ..., pn'}
    vector<Point3f> pts2_3d;  // {P2}_n, P  = {p1 , p2 , ..., pn }

    Timer t3 = chrono::steady_clock::now();
    for(DMatch m : goodMatches){  // Loop through feature matches
        // Gets the depth value of the feature points p1_i and p2_i
        // ushort: unsigned short int, [0 to 65,535]
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];  
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints2[m.trainIdx].pt.y))[int(keypoints2[m.trainIdx].pt.x)];

        // Ignores bad feature pixels
        if(d1 == 0 || d2 == 0)  // Invalid depth value
            continue;

        // Converts uint16 data to meters
        float dd1 = float(d1) / 5000.0;  // ScalingFactor from TUM Dataset.
        float dd2 = float(d2) / 5000.0;  // ScalingFactor from TUM Dataset.

        // Calculates the 3D Points by using the 2D Normalized Coordinates and the Depth Values
        // x = [x, y] = [X/Z, Y/Z]
        Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
        Point2f x2 = pixel2cam(keypoints2[m.trainIdx].pt, K);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2

        pts1_3d.push_back(Point3f(x1.x * dd1, x1.y * dd1, dd1));  // {P1}_n, P1 = [X1, Y1, Z1]^T = [x*Z, y*Z, Z]^T
        pts2_3d.push_back(Point3f(x2.x * dd2, x2.y * dd2, dd2));  // {P2}_n, P2 = [X2, Y2, Z2]^T = [x*Z, y*Z, Z]^T
    }
    Timer t4 = chrono::steady_clock::now();
    
    //--- Step 2.1: Iterative Closest Point (ICP) - SVD
    Mat R, t;  // Rotation Matrix, Translation Vector

    Timer t5 = chrono::steady_clock::now();
    ICP_SVD(pts1_3d, pts2_3d, R, t);
    Timer t6 = chrono::steady_clock::now();
    
    //--- Step 2.2: Iterative Closest Point (ICP) - Bundle Adjustment (BA)
    Timer t7 = chrono::steady_clock::now();
    ICP_bundleAdjustment(pts1_3d, pts2_3d, R, t);
    Timer t8 = chrono::steady_clock::now();
    
    /* --------- */
    /*  Results  */
    /* --------  */
    cout << "-----------------------------------------------------" << endl;
    printElapsedTime("Pose estimation 3D-3D: ", t1, t8);
    printElapsedTime(" | Features Calculation: ", t1, t2);
    printElapsedTime(" | Create 3D-3D Pairs: ", t3, t4);
    printElapsedTime(" | Iterative Closest Point (ICP, SVD): ", t5, t6);
    printElapsedTime(" | Iterative Closest Point (ICP, BA): ", t7, t8);
    
    // NOTE: Observe that not all the 79 feature matches have valid depth values. 7 3D-3D pairs were discarded.
    cout << "\n-- Number of 3D-3D pairs: " << pts1_3d.size() << endl;
    cout << "-----------------------------------------------------" << endl;

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

/* "Readers can compare R, t solved in the previous 2D-2D case to see the difference.
It can be seen that when 3D information is involved, the estimated R is almost the
same, while the t is quite different." */

/* =========================================== */
/*  Results from Pose Estimation 2D-2D (Nick)  */
/* =========================================== */
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

/* =========================================== */
/*  Results from Pose Estimation 3D-2D (Nick)  */
/* =========================================== */
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

/* ======================================================================== */
/*  Results from Pose Estimation 3D-2D (XiangGao, pixel2cam using Point2f)  */
/* ======================================================================== */
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

/* ======================================================================== */
/*  Results from Pose Estimation 3D-2D (XiangGao, pixel2cam using Point2d)  */
/* ======================================================================== */
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

/* ==================================================================== */
/*  Results from Pose Estimation 3D-3D (Nick, pixel2cam using Point2f)  */
/* ==================================================================== */
// Centroids:
// -- p1_cent: [-0.0719497, -0.102462, 1.65064]
// -- p2_cent: [-0.127945, -0.0629768, 1.68894]

/* ----- SVD Decomposition ----- */
// W: 
//   10.871 -1.01948  2.54771
// -2.16033  3.85307 -5.77742
//  3.94738 -5.79979  9.62203
// (3, 3)

// U: 
//   0.558087  -0.829399 -0.0252034
//  -0.428009  -0.313755   0.847565
//   0.710878   0.462228   0.530093
// (3, 3)

// V: 
//   0.617887  -0.784771 -0.0484806
//  -0.399894  -0.366747   0.839989
//   0.676979   0.499631   0.540434
// (3, 3)

/* ----- ICP via SVD results ----- */
// R, R12:
// [0.9969452349200465, 0.05983347846666433, -0.05020112971629315;
//  -0.05932607711456848, 0.9981719677231771, 0.01153860588791605;
//  0.05079975535776821, -0.008525122064790724, 0.9986724724099343]
// (3, 3)

// t, t12:
// [0.1441598539458168;
//  -0.06667852731466012;
//  -0.03009747291721965]
// (3, 1)

// R_inv, R21:
// [0.9969452349200465, -0.05932607711456848, 0.05079975535776821;
//  0.05983347846666433, 0.9981719677231771, -0.008525122064790724;
//  -0.05020112971629315, 0.01153860588791605, 0.9986724724099343]
// (3, 3)

// t_inv, t21:
// [-0.1461463106503255;
//  0.05767446666727236;
//  0.03806388246721711]
// (3, 1)

/* ----- ICP via g2o results ----- */
// T*, T12*: 
//    0.996945   0.0598335  -0.0502011     0.14416
//  -0.0593261    0.998172   0.0115386  -0.0666785
//   0.0507998 -0.00852511    0.998672  -0.0300979
//           0           0           0           1
// (4, 4)

/* ======================================================================== */
/*  Results from Pose Estimation 3D-3D (XiangGao, pixel2cam using Point2d)  */
/* ======================================================================== */
// Centroids: 
// [-0.0719497, -0.102462, 1.65064]
// [-0.127945, -0.0629768, 1.68894

/* ----- SVD Decomposition ----- */
// W=  10.871 -1.01948  2.54771
// -2.16033  3.85307 -5.77742
//  3.94738 -5.79979  9.62203
// 
// U=  0.558087  -0.829399 -0.0252034
//  -0.428009  -0.313755   0.847565
//   0.710878   0.462228   0.530093
// 
// V=  0.617887  -0.784771 -0.0484806
//  -0.399894  -0.366747   0.839989
//   0.676979   0.499631   0.540434

/* ----- ICP via SVD results ----- */
// R = [0.9969452349468715, 0.05983347698056557, -0.05020113095482046;
//  -0.05932607657705309, 0.9981719679735133, 0.01153858699565957;
//  0.05079975545906246, -0.008525103184062521, 0.9986724725659557]
// t = [0.144159841091821;
//  -0.06667849443812729;
//  -0.03009747273569774]
// 
// R_inv = [0.9969452349468715, -0.05932607657705309, 0.05079975545906246;
//  0.05983347698056557, 0.9981719679735133, -0.008525103184062521;
//  -0.05020113095482046, 0.01153858699565957, 0.9986724725659557]
// t_inv = [-0.1461462958593589;
//  0.05767443542067568;
//  0.03806388018483625

/* ----- ICP via g2o results ----- */
// T=
//   0.996945  0.0598335 -0.0502011    0.14416
// -0.0593261   0.998172  0.0115386 -0.0666785
//  0.0507998 -0.0085251   0.998672 -0.0300979
//          0          0          0          1