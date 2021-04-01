/* Custom Libraries */
#include "../include/stereoVision.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
string left_filepath = "../../stereoVision/src/left.png";
string right_filepath = "../../stereoVision/src/right.png";

// Stereo Camera Intrinsics params
double fx = 718.856,  fy = 718.856;   // Focal lengths
double cx = 607.1928, cy = 185.2157;  // Optical Centers

double b = 0.573;                     // Baseline

/* ====== */
/*  Main  */
/* ====== */
/* Now we start from the left and right images, calculate the disparity map corresponding to the left eye, and
then calculate the coordinates of each pixel in the camera coordinate system, which will form a point cloud */
int main(int argc, char **argv){
    cout << "[stereoVision] Hello!" << endl << endl;

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
    printImageInfo("left", left);
    printImageInfo("right", right);

    // 2. Stereo Matching (Pixel Correspondence)
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);  // SGBM is sensitive to parameters
    cv::Mat disparity_sgbm, disparity;

    cout << "[stereoVision] Computing SGBM (Semi-Global Block Matching) Disparity Map..." << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    sgbm->compute(left, right, disparity_sgbm);  // Outputs a 32-bit Disparity Map
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>> (t2-t1);
    cout << "time elapsed: " << time_elapsed.count() << " s" << endl << endl;

    // SGBM returns a Disparity Map that uses 1/16 increments.
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0/16.0f);  //So to get the true disparity it's necessary to divided by 16.

    printImageInfo("disparity_sgbm", disparity_sgbm);
    printImageInfo("disparity", disparity);

    // 3. Compute the point cloud
    PointCloud pointcloud;  // Vector of 4D Points

    // Change v++ and u++ to v+=2, u+=2, if your machine is slow to get a sparser cloud
    for (int v=0; v < left.rows; v++)
        for (int u=0; u < left.cols; u++){
            // If disparity (d) is in (-inf, 0.0] U [96.0, +inf).
            // These are the numDisparitiesMin and numDisparitiesMax params of SGBM
            float d = disparity.at<float>(v, u);
            if(d <= 0.0 || d >= 96.0)
                continue;  // Skips current iteration

            // Compute the Depth from the disparity values
            // P = ~Pc = [X, Y, Z]', P described in the Camera System
            // Pc = [X/Y, Y/Z, 1]', Normalized Coordinates


            // 1. Create a 4D Vector for holding the Camera 3D Point + Color (GreyScale)
            // point = [X, Y, Z, I(v,u)]'
            Vector4d point(0, 0, 0, left.at<uchar>(v, u)/255.0);  // Normalizes the Pixel Intensities for displaying in Pangolin's Point Cloud Viewer.


            // 2. Pixel, p=Puv=[u,v]' -> Normalized, x=Pc=[X/Z, Y/Z, 1]'
            double x = (u - cx)/fx;                       // x = X/Z
            double y = (v - cy)/fy;                       // y = Y/Z
            double Z = fx*b/d;                            // Z of P (Depth)

            // 3. Normalized, Pc=[X/Z, Y/Z, 1]' -> P Coordinates, ~Pc=[X, Y, Z]'
            point[0] = x*Z;                               // X of P
            point[1] = y*Z;                               // Y of P
            point[2] = Z;

            pointcloud.push_back(point);
        }

    // Normalizes the Disparity values to ~[0, 1] for displaying it in cv::imshow().
    cv::Mat disparity_norm = disparity.clone();
    cv::normalize(disparity/96.0, disparity_norm, 0, 1, cv::NORM_MINMAX);

    printImageInfo("disparity_norm", disparity_norm);

    // 4. Display Images
    cout << "[stereoVision] Displaying OpenCV images..." << endl;
    cv::imshow("left", left);
    cv::imshow("right", right);
    cv::imshow("disparity", disparity_norm);
    cv::waitKey(1);  // 1 ms (Non-Blocking)

    // 5. Display Point Cloud (Pangolin)
    cout << "[stereoVision] Initializing Pangolin's Point Cloud Viewer..." << endl;
    cout << "\nPress 'ESC' to exit the program..." << endl;
    showPointCloud(pointcloud);

    cv::destroyAllWindows();
    cout << "Done." << endl;
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