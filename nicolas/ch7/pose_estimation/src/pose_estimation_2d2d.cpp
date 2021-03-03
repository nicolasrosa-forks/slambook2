#define OPENCV3  // If not defined, OpenCV2

/* Libraries */
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../orb_features/src/1.png";
string image2_filepath = "../../orb_features/src/2.png";

double matches_lower_bound = 30.0;

/* ================= */
/*  Functions Scope  */
/* ================= */
void find_features_matches(
    const Mat &image1,
    const Mat &image2,
    vector<KeyPoint> &keypoints1,
    vector<KeyPoint> &keypoints2,
    vector<DMatch> &goodMatches);

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

     /* --- Step 5: Visualize the Matching result */
    Mat image_goodMatches;

    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches);

    /* Display */
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("image_goodMatches", image_goodMatches); 

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
    #else
        cout << "'OpenCV2' selected." << endl << endl;
        Ptr<FeatureDetector> detector = FeatureDetector::create ("ORB" );
        Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ("ORB" );
    #endif

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //--- Step 1: Detect the position of the Oriented FAST keypoints (Corner Points)
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    //--- Step 2: Calculate the BRIEF descriptors based on the position of Oriented FAST keypoints
    Timer t2 = chrono::steady_clock::now();
    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);
    Timer t3 = chrono::steady_clock::now();
    
    printTimeElapsed("ORB Features Extraction: ", t1, t3);
    printTimeElapsed(" | Oriented FAST Keypoints detection: ", t1, t2);
    printTimeElapsed(" | BRIEF descriptors calculation: ", t2, t3);

    cout << "\n-- Number of detected keypoints1: " << keypoints1.size() << endl;
    cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

    //cout << descriptors1 << endl;
    //cout << descriptors2 << endl;    

    //Mat outImage1, outImage2;
    //drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    //drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    //--- Step 3: Match the BRIEF descriptors of the two images using the Hamming distance
    vector<DMatch> matches;

    t1 = chrono::steady_clock::now();
    //BFMatcher matcher (NORM_HAMMING);
    matcher->match(descriptors1, descriptors2, matches);
    t2 = chrono::steady_clock::now();

    printTimeElapsed("ORB Features Matching: ", t1, t2);

    cout << "-- Number of matches: " << matches.size() << endl;

    //--- Step 4: Select correct matching (filtering)
    // Calculate the min & max distances
    double min_dist = 10000, max_dist = 0;

    // Find the mininum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
    for (int i = 0; i < descriptors1.rows; i++){
        double dist = matches[i].distance;
        if(dist<min_dist) min_dist = dist;
        if(dist>max_dist) max_dist = dist;
    }

    printf("-- Min dist: %f \n", min_dist);
    printf("-- Max dist: %f \n\n", max_dist);

    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching as wrong.
    // But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
    t1 = chrono::steady_clock::now();
    for (int i=0; i<descriptors1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
        }
    }
    t2 = chrono::steady_clock::now();

    printTimeElapsed("ORB Features Filtering: ", t1, t2);
    cout << "-- Number of good matches: " << goodMatches.size() << endl;

}