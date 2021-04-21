/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <exception>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../common/libUtils_basic.h"
#include "../../common/libUtils_eigen.h"
#include "../../common/libUtils_opencv.h"
#include "../include/optical_flow.h"

using namespace std;
using namespace cv;

/* Global Variables */
// string filename = "/home/nicolas/Downloads/Driving_Downtown_-_New_York_City_4K_-_USA_360p.mp4";
string filename = "/home/nicolas/Downloads/Driving_Downtown_-_San_Francisco_4K_-_USA_720p.mp4";

int nfeatures = 500;

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
template<typename T>
std::vector<T> create_copy(std::vector<T> const &vec){
    std::vector<T> v(vec);
    return v;
}

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv_video] Hello!" << endl << endl;

    /* Load the images */
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap(filename); 
    
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV3
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;

    /* Initialization */
    Mat image1_rgb, image2_rgb;
    Mat image1_grey, image2_grey;
    vector<KeyPoint> kps1;
    vector<Point2f> pts1_2d;
    
    // Optical Flow Variables
    Ptr<GFTTDetector> detector = GFTTDetector::create(nfeatures, 0.01, 20);
    
    vector<KeyPoint> cv_flow_kps2;    // Estimated KeyPoints in Image 2 by Multi-Level Optical Flow
    vector<Point2f> cv_flow_pts2_2d;  // Coordinates of Tracked Keypoints in Image 2
    vector<uchar> cv_flow_status;
    vector<float> cv_flow_error;
    
    Mat cv_flow_outImage2;

    // First frame initialization
    cap >> image1_rgb;
    assert(image1_rgb.data != nullptr);

    cv::cvtColor(image1_rgb, image1_grey, COLOR_BGR2GRAY);
    detector->detect(image1_grey, kps1);
    for(auto &kp: kps1) pts1_2d.push_back(kp.pt);

    // Variables for FPS Calculation
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;

    /* ------ */
    /*  Loop  */
    /* ------ */
    while(1){
        /* Read */
        // Capture frame-by-frame
        cap >> image2_rgb;

        // If the frame is empty, break immediately
        if (image2_rgb.empty())
            break;

        // cout << "Width : " << image2_rgb.size().width << endl;
        // cout << "Height: " << image2_rgb.size().height << endl;

        /* ----- Features Extraction and Matching ----- */
        cv::cvtColor(image2_rgb, image2_grey, COLOR_BGR2GRAY);

        /* ----- Optical Flow ----- */
        cv::calcOpticalFlowPyrLK(image1_grey, image2_grey, pts1_2d, cv_flow_pts2_2d, cv_flow_status, cv_flow_error);  // Fills the pts2_2d with the corresponding keypoints tracked in Image 2.

        /* ----- Results ----- */
        drawOpticalFlow<uchar>(image2_grey, cv_flow_outImage2, pts1_2d, cv_flow_pts2_2d, cv_flow_status);

        vector<Point2f> good_new;
        for(uint i = 0; i < pts1_2d.size(); i++){
            // Select good points
            if(cv_flow_status[i] == 1) {
                good_new.push_back(cv_flow_pts2_2d[i]);
            }
        }

        // Display
        // // imshow( "Frame1", image1);
        // // imshow( "Frame2", image2);
        imshow("image2_rgb", image2_rgb);
        imshow("image2_grey", image2_grey);
        imshow("Tracked by OpenCV", cv_flow_outImage2);

        /* ----- End Iteration ----- */
        // Next Iteration Prep
        image1_grey = image2_grey.clone();  // Save last frame
        pts1_2d = good_new;

        // Free vectors
        cv_flow_kps2.clear();
        cv_flow_pts2_2d.clear();
        cv_flow_status.clear();
        
        // FPS Calculation
        frameCounter++;
        std::time_t timeNow = std::time(0) - timeBegin;

        if (timeNow - tick >= 1){
            tick++;
            cout << "FPS: " << frameCounter << endl;
            frameCounter = 0;
        }

        // Press 'ESC' on keyboard to exit.
        char c = (char) waitKey(25);
        if(c==27) break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    cout << "Done." << endl;

    return 0;
}