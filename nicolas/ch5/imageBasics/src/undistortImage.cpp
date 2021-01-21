/* System Libraries */
#include <iostream>
// #include <fstream>
// #include <unistd.h>
// #include <iomanip>      // std::fixed, std::setprecision
#include <chrono>
#include <math.h>          // pow


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
bool debug = false;
string image_path = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch5/imageBasics/src/distorted.png";


// Rad-Tan model params
double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;

// Camera intrinsics
double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

/* Function Scopes */
void printImageShape(const cv:: Mat image){
    cout << "(" << image.rows << "," << image.cols << "," << image.channels() << ")" << endl; // (Height, Width, Channels)
}

int checkImage(const cv::Mat image){
    // Check if the data is correctly loaded
    if (image.data == nullptr) { 
        cerr << "File doesn't exist." << endl;
        return 0;
    } else{
        cout << "Successful." << endl;
    }

    // Check image type
    if (image.type()!= CV_8UC1 && image.type() != CV_8UC3){
        // We need grayscale image or RGB image
        cout << "Image type incorrect!" << endl;
        return 0;
    }

    return 1;
}

/* In this program we implement the undistortion by ourselves rather than using OpenCV */
int main(int argc, char **argv){
    print("[undistortImage] Hello!");
    
    // Read the image
    cout << "Reading '" << image_path << "'...";
    cv::Mat image;
    image = cv::imread(image_path, 0); // The image type is CV_8UC1
    

    if(!checkImage(image)){
        return 0;
    }

    // Print some basic information
    printImageShape(image);

    // Declaration of the undistorted image
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistorted = cv::Mat(rows, cols, CV_8UC1);

    // Compute the pixels for the undistorted one
    // Remember: The Distortion models need to be applied to the normalized coordinates, so:
    // 1. First, convert the pixels coordinates of 'image' to normalized coordinates [x, y, 1]'.
    // 2. Apply the Distortion Models and convert them back to pixel coordinates.
    // 3. Update the 'image_undistorted' with the new undistorted pixel values.
    for(int v=0; v<rows; v++){
        for(int u=0; u<cols; u++){
            // Note we are computing the pixel of (u,v) of the undistorted image
            // According to the rad-tan model, compute the coordinates in the distorted image
            // Step 1
            double x = (u - cx)/fx, y = (v-cy)/fy;
            double r = sqrt(pow(x,2) + pow(y,2));

            double x_distorted = x*(1+k1*pow(r,2)+k2*pow(r,4))+2*p1*x*y+p2*(pow(r,2)+2*pow(x,2));
            double y_distorted = y*(1+k1*pow(r,2)+k2*pow(r,4))+2*p2*x*y+p1*(pow(r,2)+2*pow(y,2));

            double u_distorted = fx*x_distorted + cx;
            double v_distorted = fy*y_distorted + cy;
            
            if(debug){
                cout << "u,v=(" << u << "," << v << ")" << "\tx,y=(" << x << "," << y << ")" << endl;
                cout << "u_dist,v_dist=(" << u_distorted << "," << v_distorted << ")" << "\tx_dist,y_dist=(" << x_distorted << "," << y_distorted << ")" << endl << endl;
            
                cout << "Press Enter to Continue..." << endl;
                cin.ignore();
            }

            // Check if the computed pixel is in the image boarder
            image_undistorted.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows){
                image_undistorted.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            }else{
                image_undistorted.at<uchar>(v, u) = 0;
            }

            
        }
    }

    // Display
    cv::imshow("image (distorted)", image);
    cv::imshow("image (undistorted)", image_undistorted);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "\nDone." << endl;
    return 0;
}

/* =========== */
/*  Functions  */
/* =========== */
