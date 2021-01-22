/* System Libraries */
#include <iostream>
#include <chrono>
#include <math.h>          // pow

/* OpenCV Library */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;

/* Global Variables */
bool debug = false;
string image_path = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch5/imageBasics/src/distorted.png";

// Rad-Tan model params
double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;

// Camera intrinsics params
double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

/* Function Scopes */

/* In this program we implement the undistortion by ourselves rather than using OpenCV */
int main(int argc, char **argv){
    print("[undistortImage] Hello!");
    
    // 1. Read the image as 8UC1 (Grayscale)
    cout << "[undistortImage] Reading '" << image_path << "'...";
    cv::Mat image = cv::imread(image_path, 0);  // The image type is CV_8UC1

    if(!checkImage(image)){
        return 0;
    }

    // Print some basic information
    printImageInfo(image);

    // 2. Declaration of the undistorted image
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistorted = cv::Mat(rows, cols, CV_8UC1);

    // 3. Compute the pixels for the undistorted one
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

            // Step 2
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

            // Step 3
            // Check if the computed pixels are beyond the image edges
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows){
                image_undistorted.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            }else{
                image_undistorted.at<uchar>(v, u) = 0;
            }            
        }
    }

    // 4. Display
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
