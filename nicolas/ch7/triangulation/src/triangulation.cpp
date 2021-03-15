/* =========== */
/*  Libraries  */
/* =========== */
#define OPENCV3  // If not defined, OpenCV2

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
#include "../../include/libUtils.h"
#include "../../include/find_features_matches.h"
#include "../../include/pose_estimation_2d2d.h"

using namespace std;
using namespace cv;

/* Global Variables */
// string image1_filepath = "../../orb_features/src/1.png";
// string image2_filepath = "../../orb_features/src/2.png";
string image1_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch7/orb_features/src/1.png";
string image2_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch7/orb_features/src/2.png";

int orb_nfeatures = 500;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
Point2d principal_point(325.1, 249.7);  // Camera Optical center coordinates
double focal_length = 521.0;            // Camera focal length

/* ================= */
/*  Functions Scope  */
/* ================= */
Mat vee2hat(const Mat var);


void triangulation(
    const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,
    const vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points_3d);

// For drawing
inline cv::Scalar get_color(float depth);

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use 2D-2D feature matching to estimate camera motion. */
int main(int argc, char **argv) {
    cout << "[pose_estimation_2d2d] Hello!" << endl;

    /* Load the images */
    Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;

    /* ---------------------------------- */
    /*  Features Extraction and Matching  */
    /* ---------------------------------- */
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches, orb_nfeatures, true);
    cout << "In total, we get " << goodMatches.size() << " set of feature points." << endl << endl;

    /* ----------------------- */
    /*  Pose Estimation 2D-2D  */
    /* ----------------------- */
    //--- Step 6.1: Estimate the motion (R, t) between the two images
    Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t, K);

    //--- Step 6.2: Verify E = t^*R*scale
    Mat t_hat = vee2hat(t);

    printMatrix("t_hat:\n", t_hat);
    printMatrix("t^*R=\n", t_hat*R);

    //--- Step 6.3: Verify the Epipolar Constraint, x2^T*E*x1 = 0
    int counter = 0;
    string flag;

    for(DMatch m : goodMatches){  // For each matched pair (p1, p2)_n, do...
        // Pixel Coordinates to Normalized Coordinates, (p1, p2)_n to (x1, x2)_n
        Point2f x1 = pixel2cam(keypoints1[m.queryIdx].pt, K);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
        Point2f x2 = pixel2cam(keypoints2[m.trainIdx].pt, K);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2

        // Convert to Homogeneous Coordinates
        Mat xh1 = (Mat_<double>(3,1) << x1.x, x1.y, 1);
        Mat xh2 = (Mat_<double>(3,1) << x2.x, x2.y, 1);

        // Calculate Epipolar Constraint
        double res = ((cv::Mat)(xh2.t()*t_hat*R*xh1)).at<double>(0);

        if(res > -1e-2 && res < 1e-2){
            flag = "Ok!";
            counter++;
        }else
            flag = "Failed!";

        printf("x2^T*E*x1 = % 01.19f\t%s\n", res, flag.c_str());
    }

    cout << "\nFinal Result: " << counter << "/" << goodMatches.size() << " Features Pairs respected the Epipolar Constraint!"<< endl << endl;

    /* --------------- */
    /*  Triangulation  */
    /* --------------- */
    //--- Step 7.1: Triangulation
    /* NOTE: Specifically for this program, since the T1w (inside of triangulation function) is the identity, Rcw=I_3x3 and tcw=[0,0,0]^T). 
             Thus, the reference frame is the {cam1} frame, not the {world} frame.*/
    vector<Point3d> points_3d;  // It should be Pw = [Xw, Yw, Zw], but in fact it's P = ~Pc = [X, Y, Z]!!!
    
    // Triangulation needs to solve the following equation: s2*x2 = s1*R*x1 + t.
    triangulation(keypoints1, keypoints2, goodMatches, R, t, points_3d);

    // cout << "points_3d[0]: " << points_3d[0].x << " " << points_3d[0].y << " " << points_3d[0].z << endl << endl;

    //--- Step 7.2: Verify the reprojection relationship between the triangulation points and the feature points.
    Mat image1_triangulation = image1.clone();
    Mat image2_triangulation = image2.clone();

    Mat R1w = (Mat_<double>(3, 3) <<
        1, 0, 0,
        0, 1, 0,
        0, 0, 1);  // R1w, Rcw
    Mat t1w = (Mat_<double>(3, 1) <<
        0, 0, 0);  // t1w, tcw

    // printMatrix("R1:\n", R1);
    // printMatrix("t1:\n", t1);

    for(int i=0; i < goodMatches.size(); i++){
        /*--- Projection of the 3D Point on Image Plane 1 */
        // Describes 3D Point P in the {cam1} frame.
        Mat pt1_trans = R1w*((Mat) points_3d[i]) + t1w;  // T(R1, t1), {world}->{cam1} transformation!

        // printMatrix("pt1_trans:\n", pt1_trans);

        /* Since there is not transformation between the {world} frame and the {cam1}, 
           we can get the depth (Z) value either from the `pt1_trans` or from `points_3d[i]` */
        float depth1_1 = pt1_trans.at<double>(2, 0);
        float depth1_2 = points_3d[i].z; 
        // cout << "Compare depths: " << depth1_1 << " " << depth1_2 << endl;  // Uncomment to see the comparison. They should be equal!

        circle(image1_triangulation, keypoints1[goodMatches[i].queryIdx].pt, 2, get_color(depth1_1), 2);  // Paint p1 according to the depth value from camera 1.

        /*--- Projection of the 3D Point on Image Plane 2 */
        // Describes 3D Point P in the {cam2} frame.
        // NOTE: Since `points_3d` are described in {cam1} (because T1w is identity), we don't need to apply the T1w transformation here.
        Mat pt2_trans = R*((Mat) points_3d[i])+t;  // T(R21, t21), {cam1}->{cam2} transformation! 
        float depth2 = pt2_trans.at<double>(2, 0);
        
        circle(image2_triangulation, keypoints2[goodMatches[i].trainIdx].pt, 2, get_color(depth2), 2);    // Paint p2 according to the depth value from camera 2.

        cout << "depth1: " << depth1_1 << "\tdepth2: " << depth2 << endl;
    }

    /* --------- */
    /*  Results  */
    /* --------  */
    /* Display Images */
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("image1 (Triangulation)", image1_triangulation);
    imshow("image2 (Triangulation)", image2_triangulation);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}

/* ======================= */
/*  Functions Declaration  */
/* ======================= */
Mat vee2hat(const Mat var){
    Mat var_hat = (Mat_<double>(3,3) <<
                         0.0, -var.at<double>(2,0),  var.at<double>(1,0),
         var.at<double>(2,0),                  0.0, -var.at<double>(0,0),
        -var.at<double>(1,0),  var.at<double>(0,0),                 0.0);  // Inline Initializer

    //printMatrix("var_hat:", var_hat);

    return var_hat;
}

void triangulation(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, const Mat &R, const Mat &t, vector<Point3d> &points_3d){
    Mat T1w = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0); // R1w, t1w

    Mat T21 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)); // R21, t21
    
    printMatrix("T1w:\n", T1w);
    printMatrix("T21:\n", T21);

    //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)
    vector<Point2f> points1, points2; // (x1, x2)_n

    for(DMatch m : matches){  // For each matched pair (p1, p2)_n, do...
        // Convert pixel coordinates to camera normalized coordinates
        points1.push_back(pixel2cam(keypoints1[m.queryIdx].pt, K));  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
        points2.push_back(pixel2cam(keypoints2[m.trainIdx].pt, K));  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2
    }

    /* NOTE: This part says that returns the "World" 3D Points Coordinate. 
    /* In my understanding, the term "World" should correspond to the Pw. However, since T1w is identity, I believe it's
    /* just the 3D Coordinate of a point P described on the {cam1} frame, and T21 correspondes only to the transformation
    /* from Camera {1} to Camera {2}, since the estimated (R, t) are (R21, t21), not the (R1w, t1w).
    */   
    //--- Get World 3D Points Coordinates (in Homogeneous Coordinates)
    Mat points_4d;  // It should be Pw = [Xw, Yw, Zw], but in fact it's P = ~Pc = [X, Y, Z]!!!
    triangulatePoints(T1w, T21, points1, points2, points_4d);  // Returns 4xN array of reconstructed points in homogeneous coordinates. These points are returned in the world's coordinate system.

    //--- Convert to non-homogeneous coordinates
    for (int i=0; i<points_4d.cols; i++) {
        Mat x = points_4d.col(i);
        x /= x.at<float>(3, 0);  // Normalization, Pw = [Xw, Yw, Zw, 1] or P = [X, Y, Z, 1]
        Point3d p(
            x.at<float>(0, 0),  // Xw or X
            x.at<float>(1, 0),  // Yw or Y
            x.at<float>(2, 0)   // Zw or Z
        );
        points_3d.push_back(p);  // Pw = [Xw, Yw, Zw] or P = [X, Y, Z]
    }
}

inline cv::Scalar get_color(float depth){
    float up_th = 15.0627, low_th = 7.231;    // min: 7.231, max: 15.0627
    float th_range = up_th-low_th;

    if (depth > up_th)  depth = up_th;
    if (depth < low_th) depth = low_th;

    float color = 255*(up_th-depth)/th_range;
    return cv::Scalar(color, color, color);
}