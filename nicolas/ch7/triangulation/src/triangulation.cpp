/* =========== */
/*  Libraries  */
/* =========== */
#define OPENCV3  // If not defined, OpenCV2

/* System Libraries */
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace cv;

/* Global Variables */
// string image1_filepath = "../../orb_features/src/1.png";
// string image2_filepath = "../../orb_features/src/2.png";
string image1_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch7/orb_features/src/1.png";
string image2_filepath = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch7/orb_features/src/2.png";

double matches_lower_bound = 30.0;

// Camera Internal parameters, TUM Freiburg2
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
Point2d principal_point(325.1, 249.7);  // Camera Optical center coordinates, TUM Dataset calibration value
double focal_length = 521.0;            // Camera focal length, TUM dataset calibration value.

/* ================= */
/*  Functions Scope  */
/* ================= */
void find_features_matches(
    const Mat &image1, const Mat &image2,
    vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
    vector<DMatch> &goodMatches);

void pose_estimation_2d2d(
    const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,
    const vector<DMatch> &matches,
    Mat &R, Mat &t);

void triangulation2(
    const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,
    const vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points);

// For drawing
inline cv::Scalar get_color(float depth){
    float up_th = 50.0, low_th = 10.0, th_range = up_th-low_th;

    if (depth > up_th)  depth = up_th;
    if (depth < low_th) depth = low_th;

    return cv::Scalar(255*depth/th_range, 0, 255*(1-depth/th_range));
}

Mat vee2hat(const Mat var);

// Pixel coordinates to camera normalized coordinates
Point2f pixel2cam(const Point2d &p, const Mat &K);

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
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches);
    cout << "In total, we get " << goodMatches.size() << " set of feature points." << endl << endl;

    /* ----------------------- */
    /*  Pose Estimation 2D-2D  */
    /* ----------------------- */
    //--- Step 6.1: Estimate the motion (R, t) between the two images
    Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t);

    //--- Step 7.1: Triangulation
    vector<Point3d> points;
    // triangulation(keypoints1, keypoints2, goodMatches, R, t, points);
    triangulation2(keypoints1, keypoints2, goodMatches, R, t, points);

    cout << "points[0]: " << points[0].x << " " << points[0].y << " " << points[0].z << endl << endl;

    //--- Step 7.2: Verify the reprojection relationship between the triangulation points and the feature points.
    Mat image1_plot = image1.clone();
    Mat image2_plot = image2.clone();

    for(int i=0; i < goodMatches.size(); i++){
        // First Picture
        Point2f pt1_cam = pixel2cam(keypoints1[goodMatches[i].queryIdx].pt, K);  // p1->x1
        float depth1 = points[i].z;
        
        circle(image1_plot, keypoints1[goodMatches[i].queryIdx].pt, 2, get_color(depth1), 2);
        
        // Second Picture
        Mat pt2_trans = R*((Mat) points[i])+t;  // p2->x2
        float depth2 = pt2_trans.at<double>(2, 0);
        
        circle(image2_plot, keypoints2[goodMatches[i].trainIdx].pt, 2, get_color(depth2), 2);

        cout << "depth1: " << depth1 << "\tdepth2: " << depth2 << endl;


//        Mat pts2_trans = R* ((Mat) points[i])
        

    }

    // TODO: Terminar

    /* --------- */
    /*  Results  */
    /* --------  */
    /* Display Images */
    imshow("image1", image1);
    imshow("image2", image2);
    waitKey(0);

    cout << "\nDone." << endl;

    return 0;
}

void find_features_matches(const Mat &image1, const Mat &image2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &goodMatches){
    //--- Initialization
    Mat descriptors1, descriptors2;

    #ifdef OPENCV3
        cout << "'OpenCV3' selected." << endl << endl;
        Ptr<FeatureDetector> detector = ORB::create();
        Ptr<DescriptorExtractor> descriptor = ORB::create();
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    #else
        cout << "'OpenCV2' selected." << endl << endl;
        Ptr<FeatureDetector> detector = FeatureDetector::create ("ORB" );
        Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ("ORB" );
        BFMatcher matcher(NORM_HAMMING);
    #endif

    //--- Step 1: Detect the position of the Oriented FAST keypoints (Corner Points)
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    Timer t2 = chrono::steady_clock::now();

    //--- Step 2: Calculate the BRIEF descriptors based on the position of Oriented FAST keypoints
    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);
    Timer t3 = chrono::steady_clock::now();

    //cout << descriptors1 << endl;
    //cout << descriptors2 << endl;

    //Mat outImage1, outImage2;
    //drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    //drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    //--- Step 3: Match the BRIEF descriptors of the two images using the Hamming distance
    vector<DMatch> matches;

    Timer t4 = chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);
    Timer t5 = chrono::steady_clock::now();

    //--- Step 4: Select correct matching (filtering)
    // Calculate the min & max distances
    double min_dist = 10000, max_dist = 0;

    // Find the mininum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
    for (int i = 0; i < descriptors1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching as wrong.
    // But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
    Timer t6 = chrono::steady_clock::now();
    for (int i=0; i<descriptors1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
            // cout << matches[i].distance << endl;
        }
    }
    Timer t7 = chrono::steady_clock::now();

    //--- Step 5: Visualize the Matching result
    Mat image_goodMatches;

    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches);

    imshow("image_goodMatches", image_goodMatches);

    /* Results */
    printTimeElapsed("ORB Features Extraction: ", t1, t3);
    printTimeElapsed(" | Oriented FAST Keypoints detection: ", t1, t2);
    printTimeElapsed(" | BRIEF descriptors calculation: ", t2, t3);
    cout << "\n-- Number of detected keypoints1: " << keypoints1.size() << endl;
    cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

    printTimeElapsed("ORB Features Matching: ", t4, t5);
    cout << "-- Number of matches: " << matches.size() << endl;
    cout << "-- Min dist: " << min_dist << endl;
    cout << "-- Max dist: " << max_dist << endl << endl;

    printTimeElapsed("ORB Features Filtering: ", t6, t7);
    cout << "-- Number of good matches: " << goodMatches.size() << endl;

}

void pose_estimation_2d2d(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, Mat &R, Mat &t){
    //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)
    vector<Point2f> points1, points2;

    for (int i=0; i < (int) matches.size(); i++){
        // Convert pixel coordinates to camera normalized coordinates
        cout << i << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    cout << endl;

    //--- Calculate the Fundamental Matrix
    Timer t1 = chrono::steady_clock::now();
    Mat F = findFundamentalMat(points1, points2, CV_FM_8POINT);  // 8-Points Algorithm
    Timer t2 = chrono::steady_clock::now();

    //--- Calculate the Essential Matrix
    Mat E = findEssentialMat(points1, points2, focal_length, principal_point);  // Remember: E = t^*R = K^T*F*K, Essential matrix needs intrinsics info.
    Timer t3 = chrono::steady_clock::now();

    //--- Calculate the Homography Matrix
    //--- But the scene in this example is not flat, and then Homography matrix has little meaning.
    Mat H = findHomography(points1, points2, RANSAC, 3);
    Timer t4 = chrono::steady_clock::now();

    //--- Restore Rotation and Translation Information from the Essential Matrix, E = t^*R
    // In this program, OpenCV will use triangulation to detect whether the detected point’s depth is positive to select the correct solution.
    // This function is only available in OpenCV3!
    recoverPose(E, points1, points2, R, t, focal_length, principal_point);
    Timer t5 = chrono::steady_clock::now();

    /* Results */
    printTimeElapsed("Pose estimation 2D-2D: ", t1, t5);
    printTimeElapsed(" | Fundamental Matrix Calculation: ", t1, t2);
    printTimeElapsed(" |   Essential Matrix Calculation: ", t2, t3);
    printTimeElapsed(" |  Homography Matrix Calculation: ", t3, t4);
    printTimeElapsed(" |             Pose Recover(R, t): ", t4, t5);
    cout << endl;

    printMatrix("K:\n", K);
    printMatrix("F:\n", F);
    printMatrix("E:\n", E);
    printMatrix("H:\n", H);

    printMatrix("R:\n", R);
    printMatrix("t:\n", t);
}

void triangulation2(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, const Mat &R, const Mat &t, vector<Point3d> &points){
    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);

    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    
    printMatrix("T1:\n", T1);
    printMatrix("T2:\n", T2);

    //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)
    vector<Point2f> points1, points2;

    for(DMatch m : matches){  // For each matched pair (p1, p2)_n, do...
        // Convert pixel coordinates to camera normalized coordinates
        points1.push_back(pixel2cam(keypoints1[m.queryIdx].pt, K));
        points2.push_back(pixel2cam(keypoints2[m.trainIdx].pt, K));
    }

    // FIXME: This part says that returns the "World" 3D Points Coordinate. In my understanding, the term "World" should correspond to the Pw. However, since T1 is identity, I believe it's just the 3D Coordinate of a point P described on the Camera 1 frame, and T2 correspondes only to the transformation from Camera {1} to Camera {2}, since the estimated (R, t) are (R21, t21), not the (Rcw, tcw). 
    //--- Get World 3D Points Coordinates (in Homogeneous Coordinates)
    Mat pts_4d;
    triangulatePoints(T1, T2, points1, points2, pts_4d);  // Returns 4xN array of reconstructed points in homogeneous coordinates. These points are returned in the world's coordinate system.

    //--- Convert to non-homogeneous coordinates
    for (int i=0; i<pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);  // Normalization, Pw = [Xw, Yw, Zw, 1]
        Point3d p(
            x.at<float>(0, 0),  // Xw
            x.at<float>(1, 0),  // Yw
            x.at<float>(2, 0)   // Zw
        );
        points.push_back(p);
    }


}

Mat vee2hat(const Mat var){
    Mat var_hat = (Mat_<double>(3,3) <<
                         0.0, -var.at<double>(2,0),  var.at<double>(1,0),
         var.at<double>(2,0),                  0.0, -var.at<double>(0,0),
        -var.at<double>(1,0),  var.at<double>(0,0),                 0.0);  // Inline Initializer

    //printMatrix("var_hat:", var_hat);

    return var_hat;
}

/**
 * @brief Convert Pixel Coordinates to Normalized Coordinates (Image Plane, f=1)
 *
 * @param p Point2d in Pixel Coordinates, p=(u,v)
 * @param K Intrinsic Parameters Matrix
 * @return Point2d in Normalized Coordinates, x=(x,y)
 */
Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x-K.at<double>(0, 2)) / K.at<double>(0, 0),  // x = (u-cx)/fx
      (p.y-K.at<double>(1, 2)) / K.at<double>(1, 1)   // y = (v-cy)/fy
    );
}