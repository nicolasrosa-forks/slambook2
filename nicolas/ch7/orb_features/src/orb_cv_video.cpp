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
#include "../../../common/libUtils.h"
#include "../../include/find_features_matches.h"
// #include "../../include/pose_estimation_2d2d.h" //FIXME: You CAN'T use this function without the correct K matrix!

using namespace std;
using namespace cv;

/* Global Variables */
int orb_nfeatures = 500;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence // FIXME: I don't have the K for the current video! So I can't calculate the pose estimation!
// Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
// Point2d principal_point(325.1, 249.7);  // Camera Optical center coordinates
// double focal_length = 521.0;            // Camera focal length

/* ===================== */
/*  Function Prototypes  */
/* ===================== */


/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv] Hello!" << endl;

    #ifdef OPENCV3
        cout << "'OpenCV3' selected." << endl << endl;
    #else
        cout << "'OpenCV2' selected." << endl << endl;
    #endif

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
    // You can use the following instead for OpenCV 3
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;

    /* Initialization */
    Mat image1, image2;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;

    // First frame initialization
    cap >> image1;

    // Variables for FPS Calculation
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;

    /* ------ */
    /*  Loop  */
    /* ------ */
    while(1){
        /* ------ */
        /*  read  */
        /* ------ */
        // Capture frame-by-frame
        cap >> image2;

        // If the frame is empty, break immediately
        if (image1.empty() && image2.empty())
          break;

        /* ---------------------------------- */
        /*  Features Extraction and Matching  */
        /* ---------------------------------- */
        find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, orb_nfeatures, false);

        /* ------------------------------------------- */
        /*  Pose Estimation 2D-2D  (Epipolar Geometry) */
        /* ------------------------------------------- */
        //--- Step 6.1: Estimate the motion (R, t) between the two images
        Mat R, t;
        // pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t);  # FIXME: Different Camera, Different K!
        
        /* ------- */
        /*  Other */
        /* ------- */
        // Display
        // imshow( "Frame1", image1);
        // imshow( "Frame2", image2);

        // Next Iteration Prep
        image1 = image2.clone();  // Save last frame
        goodMatches.clear();  // Free vectors
        
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