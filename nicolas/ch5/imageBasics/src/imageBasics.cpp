/* System Libraries */
#include <iostream>
// #include <fstream>
// #include <unistd.h>
// #include <iomanip>      // std::fixed, std::setprecision
#include <chrono>

/* Eigen3 Libraries */
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Geometry>

/* Sophus Libraries */
// #include "sophus/se3.hpp"
// #include "sophus/so3.hpp"

/* Pangolin Library */
// #include <pangolin/pangolin.h>

/* OpenCV Library */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
// using namespace Eigen;

/* Global Variables */

/* Function Scopes */
void printImageShape(const cv:: Mat image){
    cout << "(" << image.cols << "," << image.rows << "," << image.channels() << ")" << endl; // (Width, Height, Channels)
}

/* This Program demonstrates the following operations: image reading, displaying, pixel vising, copying, assignment, etc */
int main(int argc, char **argv){
    print("helloImageBasics!");
    
    // Read the image
    // string image_path = argv[1];
    string image_path = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch5/imageBasics/src/dog.jpg";
    
    cout << "Reading '" << image_path << "'...";
    cv::Mat image;
    image = cv::imread(image_path); // call cv::imread to read the image from file
    
    // Check if the data is correctly loaded
    if (image.data == nullptr) { 
        cerr << "File doesn't exist." << endl;
        return 0;
    } else{
        cout << "Successful." << endl;
    }

    // Print some basic information
    printImageShape(image);
    cv::imshow("image", image);  // Use cv::imshow() to show the image
    cv::waitKey(0);              // Display and wait for a keyboard input

    // Check image type
    cout << image.type() << endl;
    if (image.type()!= CV_8UC1 && image.type() != CV_8UC3){
        // We need grayscale image or RGB image
        cout << "Image type incorrect!" << endl;
        return 0;
    }

    cout << "Done." << endl;
}

/* =========== */
/*  Functions  */
/* =========== */
