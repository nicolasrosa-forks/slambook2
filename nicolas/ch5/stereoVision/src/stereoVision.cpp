/* System Libraries */
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <unistd.h>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* OpenCV Library */
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
typedef vector<Vector4d, Eigen::aligned_allocator<Vector4d>> PointCloud;

string left_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch5/stereoVision/src/left.png";
string right_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch5/stereoVision/src/right.png";
    
// Camera Intrinsics params
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

// Stereo Camera baseline
double b = 0.573;

/* Function Scopes */
void showPointCloud(const PointCloud &pointcloud);

/* Now we start from the left and right images, calculate the disparity map corresponding to the left eye, and
then calculate the coordinates of each pixel in the camera coordinate system, which will form a point cloud */
int main(int argc, char **argv){
    print("[stereoVision] Hello!");
    
    // 1. Read the image as 8UC1
    cout << "[stereoVision] Reading '" << left_filepath << "'...";
    cv::Mat left = cv::imread(left_filepath, 0);

    if(!checkImage(left)){
        return 0;
    }

    cout << "[stereoVision] Reading '" << right_filepath << "'...";
    cv::Mat right = cv::imread(right_filepath, 0);

    if(!checkImage(right)){
        return 0;
    }

    // Print some basic information
    cout << endl;
    printImageInfo(left);
    printImageInfo(right);
    
    // 2. Stereo Matching (Pixel Correspondence)
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);  // SGBM is senstive to parameters
    cv::Mat disparity_sgbm, disparity;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cout << "\n[stereoVision] Computing SGBM (Semi-Global Block Matching) Disparity Map..." << endl;
    sgbm->compute(left, right, disparity_sgbm);  // Outputs a 32-bit Disparity Map
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>> (t2-t1);
    cout << "time elapsed: " << time_elapsed.count() << " s" << endl;

    disparity_sgbm.convertTo(disparity, CV_32F, 1.0/16.0f);

    // cout << disparity_sgbm << endl;
    // cout << "Press Enter to Continue..." << endl;
    // cin.ignore();

    double minVal, maxVal; 

    cv::minMaxLoc(disparity_sgbm, &minVal, &maxVal);
    cout << "disparity_sgbm: [" << minVal << "," << maxVal << "]";

    cv::minMaxLoc(disparity, &minVal, &maxVal);
    cout << "disparity: [" << minVal << "," << maxVal << "]";

    
    // 3. Compute the point cloud
    PointCloud pointcloud;  // Vector of 4D Points

    // Change v++ and u++ to v+=2, u+=2, if your machine is slow to get a sparser cloud
    for (int v=0; v < left.rows; v++)
        for (int u=0; u < left.cols; u++){
            if(disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0)  // d = (-inf, 0.0] U [96.0, +inf) //TODO: Why?
                continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u)/255.0);  // Point = (x, y, z, color)   // Normalizes the Pixel Intensities for displaying in Pangolin's Point Cloud.
            
            // Compute the Depth from disparity
            // P = ~Pc = [X, Y, Z]', P described in the Camera System
            // Pc = [X/Y, Y/Z, 1]', Normalized Coordinates

            // First, Pixel -> Normalized Coordinates
            double x = (u - cx)/fx;                             // x = X/Z
            double y = (v - cy)/fy;                             // y = Y/Z
            double depth = fx*b/(disparity.at<float>(v, u));  // Z

            point[0] = x*depth;                               // X of Pc
            point[1] = y*depth;                               // Y of Pc
            point[2] = depth;                                 

            pointcloud.push_back(point);
        }

    

    // 4. Display Images
    cv::imshow("left", left);
    cv::imshow("right", right);
    cv::imshow("disparity", disparity / 96.0);  // Normalizes the Disparity values to [0, 1] for displaying it in cv::imshow().
    cv::waitKey(0);

    // 5. Show the Point Cloud in Pangolin
    showPointCloud(pointcloud);

    cv::destroyAllWindows();
    cout << "\nDone." << endl;
    return 0;
}

/* =========== */
/*  Functions  */
/* =========== */
void showPointCloud(const PointCloud &pointcloud){
    if(pointcloud.empty()){
        cerr << "Point cloud is empty!" << endl;
        return;
    }
    
    // Create Pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);                            // The Depth Test is a per-sample processing operation performed after the Fragment Shader (and sometimes before). https://www.khronos.org/opengl/wiki/Depth_Test
    glEnable(GL_BLEND);                                 // If enabled, blend the computed fragment color values with the values in the color buffers. https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glEnable.xml
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Pixels can be drawn using a function that blends the incoming (source) RGBA values with the RGBA values that are already in the frame buffer (the destination values). https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glBlendFunc.xml

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );  // Object representing attached OpenGl Matrices/transforms

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    /* Loop */
    while (pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glPointSize(2);

        /* Draw PointCloud Points*/
        glBegin(GL_POINTS);
        for(auto &p: pointcloud){
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
   
        pangolin::FinishFrame();
        usleep(5000);  // sleep 5 ms
        
        if(cv::waitKey(10)==27){    // 'Esc' key to stop
            break;
        }
    }
    return;
}