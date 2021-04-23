/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>

/* OpenCV Libraries */
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"
#include "../include/optical_flow.h"
#include "../include/OpticalFlowTracker.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../images/LK1.png";
string image2_filepath = "../../images/LK2.png";

int nfeatures = 500;
bool saveResults = false;

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

    if (image1.empty() || image2.empty()){
        std::cerr << "[FileNotFoundError] imread() failed!" << std::endl;
        return -1;  // Don't let the execution continue, else imshow() will crash.
    }

    /* --------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    /* Initialization */
    vector<KeyPoint> kps1;    // Keypoints in Image 1
    vector<Point2f> pts1_2d;  // Coordinates of the Keypoints in Image 1

    /** GFTT: Good Features To Track (Shi-Tomasi method)
     * https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
     */
    Ptr<GFTTDetector> detector = GFTTDetector::create(nfeatures, 0.01, 20);
    
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, kps1);
    Timer t2 = chrono::steady_clock::now();

    // Fills the `pts1_2d` with the detected keypoints in Image 1.
    for(auto &kp: kps1) pts1_2d.push_back(kp.pt);

    /* ------------------ */
    /*  LK Flow in OpenCV */
    /* ------------------ */
    /* Initialization */
    vector<Point2f> cv_flow_pts2_2d;      // Coordinates of Tracked Keypoints in Image 2
    vector<uchar> cv_flow_status;         // uchar: unsigned char, [0, 255]
    vector<float> cv_flow_error;

    // Let's use OpenCV's flow for validation.
    Timer t3 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(image1, image2, pts1_2d, cv_flow_pts2_2d, cv_flow_status, cv_flow_error);  // Fills the pts2_2d with the corresponding keypoints tracked in Image 2.
    Timer t4 = chrono::steady_clock::now();

    /* --------------------------------------- */
    /*  Optical Flow with Gauss-Newton method  */
    /* --------------------------------------- */
    
    /* ----- Single-Layer Optical Flow ----- */
    /* Initialization */
    vector<KeyPoint> single_flow_kps2;    // Estimated KeyPoints in Image 2 by Single-Level Optical Flow
    vector<Point2f> single_flow_pts2_2d;  // Coordinates of Tracked Keypoints in Image 2
    vector<bool> single_flow_status;

    // Now let's track these keypoints in the second image
    // First use the single-layer LK in the validation picture
    Timer t5 = chrono::steady_clock::now();
    OpticalFlowSingleLevel(image1, image2, kps1, single_flow_kps2, single_flow_status);
    Timer t6 = chrono::steady_clock::now();

    /* ----- Multi-Layer Optical Flow ----- */
    /* Initialization */
    vector<KeyPoint> multi_flow_kps2;     // Estimated KeyPoints in Image 2 by Multi-Level Optical Flow
    vector<Point2f> multi_flow_pts2_2d;   // Coordinates of Tracked Keypoints in Image 2
    vector<bool> multi_flow_status;

    // Then let's test the multi-layer LK
    Timer t7 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(image1, image2, kps1, multi_flow_kps2, multi_flow_status, true, true);
    Timer t8 = chrono::steady_clock::now();

    /* --------- */
    /*  Results  */
    /* --------  */
    printElapsedTime("Feature Extraction (GFTT): ", t1, t2);
    printElapsedTime("Optical Flow: ", t3, t8);
    printElapsedTime(" | Opencv's LK Flow: ", t3, t4);
    printElapsedTime(" | Single-layer LK Flow: ", t5, t6);
    printElapsedTime(" | Multi-layer LK Flow: ", t7, t8);

    /* Draw tracked features in Image 2 */
    Mat cv_flow_outImage2, single_flow_outImage2, multi_flow_outImage2;
    for (auto &kp: single_flow_kps2) single_flow_pts2_2d.push_back(kp.pt);
    for (auto &kp: multi_flow_kps2) multi_flow_pts2_2d.push_back(kp.pt);

    drawOpticalFlow<uchar>(image2, cv_flow_outImage2, pts1_2d, cv_flow_pts2_2d, cv_flow_status);
    drawOpticalFlow<bool>(image2, single_flow_outImage2, pts1_2d, single_flow_pts2_2d, single_flow_status);
    drawOpticalFlow<bool>(image2, multi_flow_outImage2, pts1_2d, multi_flow_pts2_2d, multi_flow_status);

    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    imshow("Tracked by LK (Single-layer)", single_flow_outImage2);
    imshow("Tracked by LK (Multi-layer, Pyramid)", multi_flow_outImage2);
    imshow("Tracked by LK (OpenCV)", cv_flow_outImage2);

    if(saveResults){
        char buffer[100];
        int ret;
        
        ret = sprintf(buffer, "../src/results/optical_flow_img2_single_%ld.jpg", std::time(nullptr));
        cv::imwrite(buffer, single_flow_outImage2);

        ret = sprintf(buffer, "../src/results/optical_flow_img2_multi_%ld.jpg", std::time(nullptr));
        cv::imwrite(buffer, multi_flow_outImage2);

        ret = sprintf(buffer, "../src/results/optical_flow_img2_CV_%ld.jpg", std::time(nullptr));
        cv::imwrite(buffer, cv_flow_outImage2);
    }
    
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}