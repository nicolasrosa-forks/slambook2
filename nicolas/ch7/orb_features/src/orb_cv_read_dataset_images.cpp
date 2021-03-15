/* =========== */
/*  Libraries  */
/* =========== */
#define OPENCV3  // If not defined, OpenCV2

/* System Libraries */
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <string>
#include <system_error>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../include/libUtils.h"
#include "../include/find_features_matches.h"
#include "../include/pose_estimation_2d2d.h"

using namespace std;
using namespace cv;

/* Global Variables */
// int orb_nfeatures = 100; //FIXME: uncomment
// double matches_lower_bound = 30.0;  //FIXME: uncomment
string data_root = "/media/nicolas/nicolas_seagate/datasets/tum_rgbd/handheld_slam/rgbd_dataset_freiburg2_desk/rgb/";

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
std::vector<std::string> get_filepaths_in_path(const string path){
    std::vector<std::string> filepaths;

    DIR *dir = opendir(path.c_str());
    if(dir == NULL){
        throw std::system_error(EDOM, std::generic_category(), "[SystemError] Given Directory is invalid!");  // FIXME: EDOM?
    }

    struct dirent *entity;
    entity = readdir(dir);
    while (entity != NULL){
        // printf("%s\n", entity->d_name);
        filepaths.push_back(path + entity->d_name); // data_root + filename
        entity = readdir(dir);
    }

    closedir(dir);
    
    // Pop first two positions
    filepaths.erase(filepaths.begin());  // Remove "."
    filepaths.erase(filepaths.begin());  // Remove ".."

    // Sort 
    sort(filepaths.begin(), filepaths.end());

    return filepaths;
}

void print_string_info(string str){
    std::cout << str << endl;
    std::cout << "size: " << str.size() << endl;
    std::cout << "length: " << str.length() << endl;
    std::cout << "capacity: " << str.capacity() << endl;
    std::cout << "max_size: " << str.max_size() << endl;
}

void printStringVector(const std::vector<std::string> &image_filepaths){
    for(string filepath : image_filepaths){
        cout << filepath << endl;
        // print_string_info(filepath);
    }

    cin.ignore();
}

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
    std::vector<std::string> image_filepaths = get_filepaths_in_path(data_root);
    // printStringVector(image_filepaths);
    int n_images = image_filepaths.size();
    cout << "\nNumber of imagens: " << n_images << endl << endl;

    /* Initialization */
    Mat image1, image2;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;
    
    // First two frames initialization
    image1 = imread(image_filepaths[0], CV_LOAD_IMAGE_COLOR);

    // Variables for FPS Calculation
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;

    /* ------ */
    /*  Loop  */
    /* ------ */
    for(int i=0; i<n_images; i++){
        /* ------ */
        /*  read  */
        /* ------ */
        // Capture frame-by-frame
        // cap >> image2;
        image2 = imread(image_filepaths[i+1], CV_LOAD_IMAGE_COLOR);  // image_filepaths[i+1]

        // If the frame is empty, break immediately
        if (image1.empty() && image2.empty())
          break;

        /* ---------------------------------- */
        /*  Features Extraction and Matching  */
        /* ---------------------------------- */
        find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, false);

        /* ----------------------- */
        /*  Pose Estimation 2D-2D  */
        /* ----------------------- */
        //--- Step 6.1: Estimate the motion (R, t) between the two images
        Mat R, t;
        pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t, K);
        
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
            cout << "Frames per second: " << frameCounter << endl;
            frameCounter = 0;
        }

        // Press 'ESC' on keyboard to exit.
        char c = (char) waitKey(25);
        if(c==27) break;

        i++;
    }
 
    // When everything done, release the video capture object
    // cap.release();

    // Closes all the frames
    destroyAllWindows();

    cout << "Done." << endl;

    return 0;
}