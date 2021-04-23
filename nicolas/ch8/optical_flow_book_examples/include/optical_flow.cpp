/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <string>
#include <chrono>
//#include <dirent.h>
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

/** Single-layer Optical Flow
 * @param [in] img1 First image
 * @param [in] img2 Second image
 * @param [in] kps1 Detected keypoints in the Image 1
 * @param [in, out] kps2 Keypoints in Image 2. If empty, use initial guess in `kps1`
 * @param [out] success 
 * @param [in] inverse 
 * @param [in] has_initial_guess 
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

    /* Create Tracker Object */
    OpticalFlowTracker tracker(img1, img2, kps1, kps2, success, inverse, has_initial_guess);

    /* Run */
    parallel_for_(Range(0, kps1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

/** Multi-level Optical Flow
 * @brief Scale of pyramid is set to 2 by default, the image pyramid will be create inside the function.
 * 
 * @param [in] img1 First image (base of pyramid 1)
 * @param [in] img2 Second image (base of pyramid 2)
 * @param [in] kp1 Detected Keypoints in Image 1
 * @param [out] kp2 Detected Keypoints in Image 2
 * @param [out] success True, if a keypoint is tracked successfully
 * @param [in] inverse Set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kps1,
    vector<KeyPoint> &kps2,
    vector<bool> &success,
    bool inverse,
    bool verbose
){
    /* Parameters */
    int n_layers = 4;
    double pyramid_scale = 0.5;  // Scaling factor
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    /* Create Pyramids */
    Timer t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2;  // Image pyramids
    for(int i=0; i<n_layers; i++){
        if(i == 0){
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }else{
            Mat img1_layer, img2_layer;

            cv::resize(pyr1[i-1], img1_layer, cv::Size(pyr1[i-1].cols*pyramid_scale, pyr1[i-1].rows*pyramid_scale));
            cv::resize(pyr2[i-1], img2_layer, cv::Size(pyr2[i-1].cols*pyramid_scale, pyr2[i-1].rows*pyramid_scale));
            
            pyr1.push_back(img1_layer);
            pyr2.push_back(img2_layer);
        }
    }
    Timer t2 = chrono::steady_clock::now();

    if(verbose)
        printElapsedTime("Build Pyramid time: ", t1, t2);

    /* Coarse-to-Fine LK tracking in Pyramids */
    vector<KeyPoint> kps1_pyr, kps2_pyr;
    
    // Top Layer
    for(auto &kp: kps1){
        auto kp_top = kp;
        kp_top.pt *= scales[n_layers - 1];
        kps1_pyr.push_back(kp_top);
        kps2_pyr.push_back(kp_top);
    }

    // Other Layers
    for(int level = n_layers - 1; level >= 0; level--){  // Top-to-Bottom
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kps1_pyr, kps2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        
        if(verbose)
            cout << "Track pyr " << level << " cost time: " << time_used.count() << endl;

        /* Updates the KeyPoints Coordinates in the Pyramids */
        if(level > 0){
            for(auto &kp: kps1_pyr)       // Note that we're not creating a copy of the element, by acessing it through reference (&).
                kp.pt /= pyramid_scale;  // Downscales the KeyPoints Coordinates in the Keypoint Pyramid of Image 1
            for(auto &kp: kps2_pyr)       
                kp.pt /= pyramid_scale;  // Downscales the KeyPoints Coordinates in the Keypoint Pyramid of Image 2
        }
    }
    if(verbose)
        cout << endl;

    /* Returns the computed tracked points */
    for (auto &kp: kps2_pyr)
        kps2.push_back(kp);
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

