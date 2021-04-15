/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"
#include "../../include/find_features_matches.h"
#include "../../include/pose_estimation_2d2d.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../images/1.png";
string image2_filepath = "../../images/2.png";

int nfeatures = 500;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);


/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use 2D-2D feature matching to estimate camera motion. */
int main(int argc, char **argv) {
    cout << "[pose_estimation_2d2d] Hello!" << endl << endl;

    /* Load the images */
    Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;

    /* ---------------------------------- */
    /*  Features Extraction and Matching  */
    /* ---------------------------------- */
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, nfeatures, true);

    /* ------------------------------------------- */
    /*  Pose Estimation 2D-2D  (Epipolar Geometry) */
    /* ------------------------------------------- */
    // Also known as Initialization Process for Monocular SLAM
    //--- Step 6.1: Estimate the motion (R, t) between the two images
    Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t, K);

    //--- Step 6.2: Verify E = t^*R*scale
    Mat t_hat = vee2hat(t);

    printMatrix("t^:\n", t_hat);
    printMatrix("t^*R=\n", t_hat*R);

    //--- Step 6.3: Verify the Epipolar Constraint, x2^T*E*x1 = 0
    int counter = 0;
    string flag;

    for(DMatch m : goodMatches){  // For each matched pair {(p1, p2)}_n, do...
        // Pixel Coordinates to Normalized Coordinates, {(p1, p2)}_n to {(x1, x2)}_n
        Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
        Point2f x2 = pixel2cam(keypoints2[m.trainIdx].pt, K);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2

        // Convert to Homogeneous Coordinates
        Mat xh1 = (Mat_<double>(3,1) << x1.x, x1.y, 1);
        Mat xh2 = (Mat_<double>(3,1) << x2.x, x2.y, 1);

        // Calculate Epipolar Constraint
        double res = ((cv::Mat)(xh2.t()*t_hat*R*xh1)).at<double>(0);

        if(res > -1e-2 && res < 1e-2){
            flag = "Ok!";
            counter++;
        }else
            flag = "Failed!";

        printf("x2^T*E*x1 = % 01.19f\t%s\n", res, flag.c_str());
    }

    cout << "\nFinal Result: " << counter << "/" << goodMatches.size() << " Features Pairs respected the Epipolar Constraint!"<< endl << endl;

    /* --------- */
    /*  Results  */
    /* --------  */
    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}