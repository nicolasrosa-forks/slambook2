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
#include "../include/OpticalFlowTracker.h"

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
 * @param [in] has_initial_guess 
 */
void OpticalFlowSingleLevel(
    const Mat &img1, const Mat &img2, 
    const vector<KeyPoint> &kps1, vector<KeyPoint> &kps2, 
    vector<bool> &success, 
    bool inverse = false, bool has_initial_guess = false){
    
    /* Resize vectors */
    kps2.resize(kps1.size());
    success.resize(kps1.size());

    /* Create Tracker Object */
    OpticalFlowTracker tracker(img1, img2, kps1, kps2, success, inverse, has_initial_guess);

    /* Run */
    parallel_for_(Range(0, kps1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

template <typename TType>
void drawOpticalFlow(const Mat &inImage, Mat &outImage, const vector<Point2f> &pts1_2d, const vector<Point2f> &pts2_2d, vector<TType> &status){
    cv::cvtColor(inImage, outImage, COLOR_GRAY2BGR);
    
    for(int i = 0; i < pts2_2d.size(); i++){
        if(status[i]){
            cv::circle(outImage, pts2_2d[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(outImage, pts1_2d[i], pts2_2d[i], cv::Scalar(0, 250, 0));
        }
    }
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
    vector<Point2f> pts1_2d;
    vector<KeyPoint> kps1;  // Keypoints on Image 1

    /* --------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    /** GFTT: Good Features To Track (Shi-Tomasi method)
     * https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
     */
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, kps1);
    Timer t2 = chrono::steady_clock::now();

    // Fills the `pts1_2d` with the detected keypoints in Image 1.
    for (auto &kp: kps1) pts1_2d.push_back(kp.pt);

    /* ------------------ */
    /*  LK Flow in OpenCV */
    /* ------------------ */
    // Let's use OpenCV's flow for validation.
    vector<Point2f> cv_flow_pts2_2d;
    vector<uchar> cv_flow_status;  // uchar: unsigned char, [0, 255]
    vector<float> cv_flow_error;

    Timer t3 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(image1, image2, pts1_2d, cv_flow_pts2_2d, cv_flow_status, cv_flow_error);  // Fills the pts2_2d with the corresponding keypoints tracked in Image 2.
    Timer t4 = chrono::steady_clock::now();

    /* --------------------------------------- */
    /*  Optical Flow with Gauss-Newton method  */
    /* --------------------------------------- */
    /* ----- Single-Layer Optical Flow ----- */
    // Now let's track these keypoints in the second image
    vector<KeyPoint> single_flow_kps2;
    vector<Point2f> single_flow_pts2_2d;
    vector<bool> single_flow_status;

    OpticalFlowSingleLevel(image1, image2, kps1, single_flow_kps2, single_flow_status);

    /* ----- Multi-Level Optical Flow ----- */
    //TODO
    

    /* --------- */
    /*  Results  */
    /* --------  */
    Mat cv_flow_outImage2, single_flow_outImage2, multi_flow_outImage2;

    // Draw tracked features on Image 2
    for (auto &kp: single_flow_kps2) single_flow_pts2_2d.push_back(kp.pt);

    drawOpticalFlow<uchar>(image2, cv_flow_outImage2, pts1_2d, cv_flow_pts2_2d, cv_flow_status);
    drawOpticalFlow<bool>(image2, single_flow_outImage2, pts1_2d, single_flow_pts2_2d, single_flow_status);

    
    printElapsedTime(" | Feature Extraction (GFTT): ", t1, t2);
    printElapsedTime(" | Opencv's LK Flow: ", t3, t4);

    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    imshow("Tracked by Single-layer", single_flow_outImage2);
    // imshow("Tracked by Multi-level", outImage2_flow3_multi);
    imshow("Tracked by OpenCV", cv_flow_outImage2);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}