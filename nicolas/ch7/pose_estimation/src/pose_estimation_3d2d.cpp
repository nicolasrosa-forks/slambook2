/* =========== */
/*  Libraries  */
/* =========== */
#define OPENCV3  // If not defined, OpenCV2

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
#include "../../../common/libUtils.h"
#include "../../include/find_features_matches.h"
#include "../../include/pose_estimation_2d2d.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../orb_features/src/1.png";
string image2_filepath = "../../orb_features/src/2.png";
string depth1_filepath = "../../orb_features/src/1_depth.png";
string depth2_filepath = "../../orb_features/src/2_depth.png";

int orb_nfeatures = 500;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

/* ===================== */
/*  Function Prototypes  */
/* ===================== */

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use 2D-2D feature matching to estimate camera motion. */
int main(int argc, char **argv) {
    cout << "[pose_estimation_3d2d] Hello!" << endl;

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
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, orb_nfeatures, true);
    cout << "In total, we get " << goodMatches.size() << " set of feature points." << endl << endl;

    /* ------------------------------------------- */
    /*  Pose Estimation 3D-2D  (???) */ // FIXME:
    /* ------------------------------------------- */
    /* Load the depth images */
    // The depth image is a 16-bit unsigned number, single channel image (16UC1)
    Mat d1 = imread(depth1_filepath, CV_LOAD_IMAGE_UNCHANGED);  
    Mat d2 = imread(depth2_filepath, CV_LOAD_IMAGE_UNCHANGED);

    // For plotting
    Mat d1_uint8 = imread(depth1_filepath, CV_LOAD_IMAGE_GRAYSCALE);  
    Mat d2_uint8 = imread(depth2_filepath, CV_LOAD_IMAGE_GRAYSCALE);

    // Create 3D-2D pairs
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    // Loop through feature matches
    for(DMatch m : goodMatches){
        // Gets the depth value of the feature point p_i
        ushort d = d1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];  // ushort: unsigned short int, [0 to 65,535]

        // Discards invalid feature pixels
        if(d == 0)  // bad depth
            continue;

        // TUM Dataset, Converts uint16 data to meters
        float dd = d / 5000.0;

        // Calculates the 3D Points, // FIXME: In {world} frame?
        // x = [X/Z, Y/Z]
        Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1

        pts_3d.push_back(Point3f(x1.x * dd, x1.y * dd, dd));  // Pw = [Xw, Yw, Zw]^T // FIXME: In {World} frame?
        pts_2d.push_back(keypoints2[m.trainIdx].pt);          // (p2)_n
    }
    
    // NOTE: Observe that not all the 79 feature matches have valid depth values. 4 3D-2D pairs were discarded.
    cout << "Number of 3D-2D pairs: " << pts_3d.size() << endl;

    /* --------- */
    /*  Results  */
    /* --------  */
    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    imshow("depth1", d1_uint8);
    imshow("depth2", d2_uint8);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}

/* ======================= */
/*  Functions Declaration  */
/* ======================= */