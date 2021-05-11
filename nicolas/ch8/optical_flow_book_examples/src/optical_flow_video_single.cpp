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
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"
#include "../../../common/libUtils_fps.h"
#include "../include/optical_flow.h"

using namespace std;
using namespace cv;

/* Global Variables */
// Choose:
string filename = "/home/nicolas/Downloads/Driving_Downtown_-_New_York_City_4K_-_USA_360p.mp4";
// string filename = "/home/nicolas/Downloads/Driving_Downtown_-_San_Francisco_4K_-_USA_720p.mp4";

int nfeatures = 500;
int min_nfeatures = 250;

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
int main(int argc, char **argv) { // FIXME: Acho que não está funcionando corretamente.
    cout << "[orb_cv_video] Hello!" << endl << endl;

    /* Load the images */
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap(filename);  // Create a VideoCapture object and open the input file

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Get Original FPS from the video
    double fps_v = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps_v << endl;

    /* Initialization */
    Mat image1_bgr, image2_bgr;
    Mat image1, image2;
    vector<KeyPoint> kps1;
    vector<Point2f> pts1_2d;
    
    // Optical Flow Variables
    Ptr<GFTTDetector> detector = GFTTDetector::create(nfeatures, 0.01, 20);
    
    vector<KeyPoint> single_flow_kps2;    // Estimated KeyPoints in Image 2 by Single-Level Optical Flow
    vector<Point2f> single_flow_pts2_2d;  // Coordinates of Tracked Keypoints in Image 2
    vector<bool> single_flow_status;

    Mat single_flow_outImage2;

    // Get first frame
    cap >> image1_bgr;
    assert(image1_bgr.data != nullptr);  // FIXME: I think this its not working!

    cv::cvtColor(image1_bgr, image1, COLOR_BGR2GRAY);
    detector->detect(image1, kps1);
    for(auto &kp: kps1) pts1_2d.push_back(kp.pt);

    // Variables for FPS Calculation
    FPS fps = FPS();

    /* ------ */
    /*  Loop  */
    /* ------ */
    while(1){
        /* Read */
        // Capture frame-by-frame
        cap >> image2_bgr;

        // If the frame is empty, break immediately
        if (image2_bgr.empty())
            break;

        // cout << "Width : " << image2_bgr.size().width << endl;
        // cout << "Height: " << image2_bgr.size().height << endl;

        /* ----- Features Extraction and Matching ----- */
        cv::cvtColor(image2_bgr, image2, COLOR_BGR2GRAY);

        /* ----- Optical Flow ----- */
        OpticalFlowSingleLevel(image1, image2, kps1, single_flow_kps2, single_flow_status, true, false);

        /* ----- Results ----- */
        for(auto &kp: single_flow_kps2) single_flow_pts2_2d.push_back(kp.pt);
        drawOpticalFlow<bool>(image2, single_flow_outImage2, pts1_2d, single_flow_pts2_2d, single_flow_status);

        vector<Point2f> good_pts2_2d;
        vector<KeyPoint> good_kps2;
        
        for(size_t i = 0; i < pts1_2d.size(); i++){
            // Select good points
            if(single_flow_status[i] == 1) {
                good_pts2_2d.push_back(single_flow_pts2_2d[i]);
                good_kps2.push_back(single_flow_kps2[i]);
            }
        }
        int n_good = good_pts2_2d.size();
        cout << n_good << "/" << nfeatures << endl;

        // Display
        // imshow("Frame1", image1);
        // imshow("Frame2", image2);
        imshow("image2_bgr", image2_bgr);
        // imshow("image2", image2);
        imshow("Tracked by Single-layer (Pyramid)", single_flow_outImage2);

        /* ----- End Iteration ----- */
        // Next Iteration Prep
        image1 = image2.clone();  // Save last frame

        if (n_good < min_nfeatures){  // Few Features, get detect more!
        // if (true){  // Few Features, get detect more!
            detector->detect(image1, kps1);
            for(auto &kp: kps1) pts1_2d.push_back(kp.pt);
            for(auto &pt: good_pts2_2d) pts1_2d.push_back(pt); // Retains previously detected keypoints
        }else{
            pts1_2d = good_pts2_2d;
        }

        // Free vectors
        // single_flow_kps2.clear();
        single_flow_pts2_2d.clear();
        single_flow_status.clear();
        
        // FPS Calculation
        fps.update();
        cout << endl;

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