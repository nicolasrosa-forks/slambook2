/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <string>

/* OpenCV Libraries */
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../common/libUtils_basic.h"
#include "../../common/libUtils_eigen.h"
#include "../../common/libUtils_opencv.h"
#include "../include/optical_flow_tracker.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../images/LK1.png";
string image2_filepath = "../images/LK2.png";

int nfeatures = 500;

/* =========== */
/*  Functions  */
/* =========== */

/** Description
 * @brief Single-level/layer Optical Flow
 * 
 * @param [in] img1 The first image
 * @param [in] img2 The second image
 * @param [in] kps1 Detected keypoints in the Image 1
 * @param [in, out] kps2 Keypoints in Image 2. If empty, use initial guess in `kps1`
 * @param [out] success 
 * @param [in] inverse 
 * @param has_initial_guess 
 */
void OpticalFlowSingleLevel(
    const Mat &img1, 
    const Mat &img2, 
    const vector<KeyPoint> &kps1, 
    vector<KeyPoint> &kps2, 
    vector<bool> &success, 
    bool inverse = false, 
    bool has_initial_guess = false){
    
    /* Resize vectors */
    kps2.resize(kps1.size());
    success.resize(kps1.size());

    /* Create Class Object */
    // OpticalFlowTracker
}

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use different techniques for calculating Optical Flow. */
int main(int argc, char **argv) {
    cout << "[optical_flow] Hello!" << endl << endl;

    /* Load the images */
    // Note, they are CV_8UC1, not CV_8UC3
    Mat image1 = imread(image1_filepath, cv::IMREAD_GRAYSCALE);
    Mat image2 = imread(image2_filepath, cv::IMREAD_GRAYSCALE);

    /* Initialization */
    vector<KeyPoint> keypoints1;

    /* -------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    /** GFTT: Good Features To Track (Shi-Tomasi method)
     * https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
     */
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, keypoints1);
    Timer t2 = chrono::steady_clock::now();

    /* ------------------ */
    /*  LK Flow in OpenCV */
    /* ------------------ */
    // Let's use opencv's flow for validation.
    vector<Point2f> pts1_2d, pts2_2d;
    vector<uchar> status;  // uchar: unsigned char, [0, 255]
    vector<float> error;

    // Fills the pts1_2d with the detected keypoints in Image 1.
    for (auto &kp: keypoints1) pts1_2d.push_back(kp.pt);

    Timer t3 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(image1, image2, pts1_2d, pts2_2d, status, error);  // Fills the pts2_2d with the corresponding keypoints tracked in Image 2.
    Timer t4 = chrono::steady_clock::now();

    /* --------------------------------------- */
    /*  Optical Flow with Gauss-Newton method  */
    /* --------------------------------------- */
    /* ----- Single-Layer Optical Flow ----- */
    // Now lets track these keypoints in the second image
    vector<KeyPoint> keypoints2_single;
    vector<bool> success_single;

    OpticalFlowSingleLevel(image1, image2, keypoints1, keypoints2_single, success_single);


    /* ----- Multi-Level Optical Flow ----- */
    //TODO
    

    /* --------- */
    /*  Results  */
    /* --------  */
    Mat outImage2_flow1_single, outImage2_flow2_multi, outImage2_flow3_opencv;

    // Draw tracked features on Image 2 - Method 1
    
    // Draw tracked features on Image 2 - Method 2

    // Draw tracked features on Image 2 - Method 3 (Opencv's LK Flow)
    cv::cvtColor(image2, outImage2_flow3_opencv, COLOR_GRAY2BGR);
    for(int i = 0; i < pts2_2d.size(); i++){
        if(status[i]){
            cv::circle(outImage2_flow3_opencv, pts2_2d[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(outImage2_flow3_opencv, pts1_2d[i], pts2_2d[i], cv::Scalar(0, 250, 0));
        }
    }

    printElapsedTime(" | Feature Extraction (GFTT): ", t1, t2);
    printElapsedTime(" | Opencv's LK Flow: ", t3, t4);

    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    // imshow("Tracked by Single-layer", outImage2_flow1_single);
    // imshow("Tracked by Multi-level", outImage2_flow3_multi);
    imshow("Tracked by OpenCV", outImage2_flow3_opencv);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}