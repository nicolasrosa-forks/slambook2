/* System Libraries */
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>      // std::fixed, std::setprecision
#include <chrono>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Sophus Libraries */
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* OpenCV Library */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Boost */
#include <boost/format.hpp>  // For formating strings

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
typedef vector<double, Eigen::aligned_allocator<double>> TimeStamp;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;  // Since Eigen doesn't have Vector6d, we need to create it.
typedef vector<Vector6d, Eigen::aligned_allocator<Vector6d>> PointCloud;

string pose_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch5/rgbd/src/pose.txt";
string rgbd_folder_path = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/ch5/rgbd/";

// RGB-D Camera Intrinsics params
double fx = 518.0, fy = 519.0;  // Focal Lengths
double cx = 325.5, cy = 253.5;  // Optical Centers
    
double depthScale = 1000.0;     // Depth Scale for decoding the Depth data.

/* Function Scopes */
TrajectoryType ReadTrajectory(TimeStamp &timestamps, const string &path);
TrajectoryType ReadTrajectory2(const string &path);
void showPointCloud(const PointCloud &pointcloud);

/* This program accomplishes two things: 
    1. Calculate the point cloud corresponding to each pair of RGB-D images based on internal parameters;
    2. According to the camera pose of each image (that is, external parameters), put the points to a global cloud by the camera poses. 
*/
int main(int argc, char **argv){
    print("[jointMap] Hello!");

    // 1. Read camera poses file
    TrajectoryType poses = ReadTrajectory2(pose_filepath);  // FIXME: File not found doesn't kill the program.

    // 2. Read RGB-D images
    vector<cv::Mat> colorImgs, depthImgs;
    for(size_t i=0; i<poses.size(); i++){  // [0, 4]
        boost::format fmt(rgbd_folder_path+"%s/%d.%s");  // the image filename format
        colorImgs.push_back(cv::imread((fmt % "color" % (i+1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i+1) % "pgm").str(), -1));  // Use -1 (IMREAD_UNCHANGED) flag to load the depth image

        boost::format fmt2("%sImgs[%d]");

        // cv::imshow((fmt2 % "color" % i).str(), colorImgs[i]);
        // cv::imshow((fmt2 % "depth" % i).str(), depthImgs[i]);

        cv::imshow("color", colorImgs[i]);
        cv::imshow("depth", depthImgs[i]);
        cv::waitKey(1);  // 1 s (Non-Blocking)
    }

    // 3. Compute the point cloud using the camera intrinsics params


    PointCloud pointcloud;
    pointcloud.reserve(1000000);  // Requests that the vector capacity be at least enough to contain n elements.

    for(size_t i=0; i<poses.size(); i++){
        cout << "[jointMap] Converting RGBD images..." << i+1 << endl;
        auto color = colorImgs[i];
        auto depth = depthImgs[i];

        Sophus::SE3d T = poses[i];  // Remember: T belongs to SE(3)

        for (int v=0; v<color.rows; v++){
            for (int u=0; u<depth.cols;u++){
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth value is 16-bit (unsigned short)
                // unsigned int d = depth.at<unsigned short>(v, u); // depth value is 16-bit (unsigned short) //FIXME: igual à instrução de cima?

                if(d==0)
                    continue;  // If d is 0 (no valid value), skips current iteration.
    
                // Compute the Depth from the RGB-D values
                // P = ~Pc = [X, Y, Z]', P described in the Camera System
                // Pc = [X/Y, Y/Z, 1]', Normalized Coordinates      

                // 1. Create a 3D Vector for holding the Camera 3D Point
                // point = [X, Y, Z]'
                Vector3d point(0,0,0);                        // P

                // 2. Pixel, Puv=[u,v]' -> Normalized, Pc=[X/Z, Y/Z, 1]'
                double x = (u - cx)/fx;                       // x = X/Z
                double y = (v - cy)/fy;                       // y = Y/Z
                double Z = double(d) / depthScale;            // Z of P (Depth)
                
                // 3. Normalized, Pc=[X/Z, Y/Z, 1]' -> P Coordinates, ~Pc=[X, Y, Z]'
                point[0] = x*Z;                               // X of P
                point[1] = y*Z;                               // Y of P
                point[2] = Z;

                // 4. Camera, ~Pc=[X, Y, Y]' -> World Coordinates, Pw = [Xw, Yw, Zw]'
                Vector3d wPoint = T*point;  // wP = Twc*P = Twc*~Pc

                // 5. Create a 6D Vector for holding the World 3D Point + Color (RGB)
                Vector6d p;
                p.head<3>() = wPoint;
                p[5] = color.data[v*color.step + u*color.channels()];      // Blue
                p[4] = color.data[v*color.step + u*color.channels() + 1];  // Green
                p[3] = color.data[v*color.step + u*color.channels() + 2];  // Red
                
                pointcloud.push_back(p);
            }
        }
    }

    // 4. Display Images
    // cv::waitKey(1);  // 1 ms (Non-Blocking)

    // 5. Display Point Cloud (Pangolin)
    showPointCloud(pointcloud);
    cv::destroyAllWindows();
    cout << "Done." << endl;
    return 0;
}

/* =========== */
/*  Functions  */
/* =========== */
TrajectoryType ReadTrajectory(TimeStamp &timestamps, const string &path){
    ifstream fin(path);  // The 'pose.txt' contains the Twc transformations!
    TrajectoryType trajectory;

    cout << "[jointMap] Reading '" << path << "'... ";
    if(!fin){
        cerr << "File not found!" << endl;
        return trajectory;
    }else{
        cout << "Successful" << endl;
    }

    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Transformation Matrix (T), Pose
        Sophus::SE3d pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));  // T, SE(3) from q,t.
        timestamps.push_back(time);
        trajectory.push_back(pose);
    }
    cout << "Read total of " << trajectory.size() << " pose entries." << endl << endl;

    return trajectory;
}

TrajectoryType ReadTrajectory2(const string &path){
    ifstream fin(path);  // The 'pose.txt' contains the Twc transformations!
    TrajectoryType trajectory;

    cout << "[jointMap] Reading '" << path << "'... ";
    if(!fin){
        cerr << "File not found!" << endl;
        return trajectory;
    }else{
        cout << "Successful" << endl;
    }

    while(!fin.eof()){
        double tx, ty, tz, qx, qy, qz, qw;
        fin >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Transformation Matrix (T), Pose
        Sophus::SE3d pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));  // T, SE(3) from q,t.
        trajectory.push_back(pose);
    }
    cout << "Read total of " << trajectory.size() << " pose entries." << endl << endl;

    return trajectory;
}

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
            glColor3f(p[3]/255.0, p[4]/255.0, p[5]/255.0);
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
