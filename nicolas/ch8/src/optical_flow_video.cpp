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
int nfeatures = 500;

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
template<typename T>
std::vector<T> create_copy(std::vector<T> const &vec){
    std::vector<T> v(vec);
    return v;
}

class myexception: public exception{
    virtual const char* what() const throw(){
        return "My exception happened";
    }
} myex;


/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv_video] Hello!" << endl << endl;

    /* Load the images */
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("/home/nicolas/Downloads/Driving_Downtown_-_New_York_City_4K_-_USA_360p.mp4"); 
    
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
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    
    vector<KeyPoint> multi_flow_kps2;    // Estimated KeyPoints in Image 2 by Multi-Level Optical Flow
    vector<Point2f> multi_flow_pts2_2d;  // Coordinates of Tracked Keypoints in Image 2
    vector<bool> multi_flow_status;
    
    Mat cv_flow_outImage2, single_flow_outImage2, multi_flow_outImage2;

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
        detector->detect(image2_grey, multi_flow_kps2);

        /* ----- Optical Flow ----- */
        OpticalFlowMultiLevel(image1_grey, image2_grey, kps1, multi_flow_kps2, multi_flow_status, true);

        /* ----- Results ----- */
        for(auto &kp: multi_flow_kps2) multi_flow_pts2_2d.push_back(kp.pt);
        drawOpticalFlow<bool>(image2_grey, multi_flow_outImage2, pts1_2d, multi_flow_pts2_2d, multi_flow_status);
        
        // Display
        // // imshow( "Frame1", image1);
        // // imshow( "Frame2", image2);
        imshow("image2_rgb", image2_rgb);
        imshow("image2_grey", image2_grey);
        imshow("Tracked by Multi-layer (Pyramid)", multi_flow_outImage2);

        /* ----- End Iteration ----- */
        // Next Iteration Prep
        image1_grey = image2_grey.clone();  // Save last frame
        pts1_2d = create_copy<Point2f>(multi_flow_pts2_2d);
        kps1 = create_copy<KeyPoint>(multi_flow_kps2);

        // Free vectors
        multi_flow_kps2.clear();
        multi_flow_pts2_2d.clear();
        multi_flow_status.clear();
        
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