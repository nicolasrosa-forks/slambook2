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

using namespace std;
using namespace cv;

/* Global Variables */
int nfeatures = 500;

/* ===================== */
/*  Function Prototypes  */
/* ===================== */


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
        /* Read */
        // Capture frame-by-frame
        cap >> image2;

        // If the frame is empty, break immediately
        if (image1.empty() && image2.empty())
            break;

        /* ----- Features Extraction and Matching ----- */
        find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, nfeatures, false);

        /* ----- Results ----- */
        // Display
        // imshow( "Frame1", image1);
        // imshow( "Frame2", image2);

        /* ----- End Iteration ----- */
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