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

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
Point2d principal_point(325.1, 249.7);  // Camera Optical center coordinates
double focal_length = 521.0;            // Camera focal length

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

Mat vee2hat(const Mat var);

// Pixel coordinates to camera normalized coordinates
Point2f pixel2cam(const Point2d &p, const Mat &K);

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
    find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches);
    cout << "In total, we get " << goodMatches.size() << " set of feature points." << endl << endl;

    /* ----------------------- */
    /*  Pose Estimation 2D-2D  */
    /* ----------------------- */
    //--- Step 6.1: Estimate the motion (R, t) between the two images
    Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t);

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

    // Find the minimum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
    for (int i = 0; i < descriptors1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching as wrong.
    // But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
//    vector<DMatch> goodMatches;

    Timer t6 = chrono::steady_clock::now();
    for (int i=0; i<descriptors1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
        }
    }
    Timer t7 = chrono::steady_clock::now();

    //--- Step 5: Visualize the Matching result
//    Mat image_matches;
    Mat image_goodMatches;

//    drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches);
    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches);

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

    /* Display */
    imshow("image_goodMatches", image_goodMatches);
}

void pose_estimation_2d2d(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, Mat &R, Mat &t){
    //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)
    vector<Point2f> points1, points2;  // (x1, x2)_n

    for (int i=0; i < (int) matches.size(); i++){  // For each matched pair (p1, p2)_n, do...
        // Convert pixel coordinates to camera normalized coordinates
        cout << i << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
        points1.push_back(keypoints1[matches[i].queryIdx].pt);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
        points2.push_back(keypoints2[matches[i].trainIdx].pt);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2
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