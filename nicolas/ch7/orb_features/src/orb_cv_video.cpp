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
int orb_nfeatures = 100;
double matches_lower_bound = 30.0;

// Camera Internal parameters, TUM Dataset Freiburg2 sequence
// Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
// Point2d principal_point(325.1, 249.7);  // Camera Optical center coordinates
// double focal_length = 521.0;            // Camera focal length

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

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv] Hello!" << endl;

    // /* Load the images */
    // Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    // Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    // assert(image1.data != nullptr && image2.data != nullptr);

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("/home/nicolas/Downloads/Driving_Downtown_-_New_York_City_4K_-_USA_360p.mp4"); 
    
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // double fps = video.get(CV_CAP_PROP_FPS);
    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV 3
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> goodMatches;

    Mat image1;
    Mat image2;

    // First two frames initialization
    cap >> image1;

    // Variables for FPS Calculation
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;

    while(1){
        /* ------ */
        /*  read  */
        /* ------ */
        // Capture frame-by-frame
        cap >> image2;
 
        // If the frame is empty, break immediately
        if (image1.empty() && image2.empty())
          break;

        /* ---------------------------------- */
        /*  Features Extraction and Matching  */
        /* ---------------------------------- */
        find_features_matches(image1, image2, keypoints1, keypoints2, goodMatches);

        /* ----------------------- */
        /*  Pose Estimation 2D-2D  */
        /* ----------------------- */
        //--- Step 6.1: Estimate the motion (R, t) between the two images
        Mat R, t;
        // pose_estimation_2d2d(keypoints1, keypoints2, goodMatches, R, t);  # FIXME: Different Camera, Different K!
        
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
    }
 
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

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
        Ptr<FeatureDetector> detector = ORB::create(orb_nfeatures);
        Ptr<DescriptorExtractor> descriptor = ORB::create(orb_nfeatures);
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

    Mat outImage1, outImage2;
    drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

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
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);
    imshow("image_goodMatches", image_goodMatches);
}

// void pose_estimation_2d2d(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches, Mat &R, Mat &t){
//     //--- Convert the Matched Feature points to the form of vector<Point2f> (Pixels Coordinates)
//     vector<Point2f> points1, points2;  // (x1, x2)_n

//     for (int i=0; i < (int) matches.size(); i++){  // For each matched pair (p1, p2)_n, do...
//         // Convert pixel coordinates to camera normalized coordinates
//         cout << i << " " << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
//         points1.push_back(keypoints1[matches[i].queryIdx].pt);  // p1->x1, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 1
//         points2.push_back(keypoints2[matches[i].trainIdx].pt);  // p2->x2, Camera Normalized Coordinates of the n-th Feature Keypoint in Image 2
//     }

//     cout << endl;

//     //--- Calculate the Fundamental Matrix
//     Timer t1 = chrono::steady_clock::now();
//     Mat F = findFundamentalMat(points1, points2, CV_FM_8POINT);  // 8-Points Algorithm
//     Timer t2 = chrono::steady_clock::now();

//     //--- Calculate the Essential Matrix
//     Mat E = findEssentialMat(points1, points2, focal_length, principal_point);  // Remember: E = t^*R = K^T*F*K, Essential matrix needs intrinsics info.
//     Timer t3 = chrono::steady_clock::now();

//     //--- Calculate the Homography Matrix
//     //--- But the scene in this example is not flat, and then Homography matrix has little meaning.
//     Mat H = findHomography(points1, points2, RANSAC, 3);
//     Timer t4 = chrono::steady_clock::now();

//     //--- Restore Rotation and Translation Information from the Essential Matrix, E = t^*R
//     // In this program, OpenCV will use triangulation to detect whether the detected point’s depth is positive to select the correct solution.
//     // This function is only available in OpenCV3!
//     recoverPose(E, points1, points2, R, t, focal_length, principal_point);
//     Timer t5 = chrono::steady_clock::now();

//     /* Results */
//     printTimeElapsed("Pose estimation 2D-2D: ", t1, t5);
//     printTimeElapsed(" | Fundamental Matrix Calculation: ", t1, t2);
//     printTimeElapsed(" |   Essential Matrix Calculation: ", t2, t3);
//     printTimeElapsed(" |  Homography Matrix Calculation: ", t3, t4);
//     printTimeElapsed(" |             Pose Recover(R, t): ", t4, t5);
//     cout << endl;

//     printMatrix("K:\n", K);
//     printMatrix("F:\n", F);
//     printMatrix("E:\n", E);
//     printMatrix("H:\n", H);

//     printMatrix("R:\n", R);
//     printMatrix("t:\n", t);
// }