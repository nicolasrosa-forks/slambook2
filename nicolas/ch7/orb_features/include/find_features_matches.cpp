/* =========== */
/*  Libraries  */
/* =========== */
#define OPENCV3  // If not defined, OpenCV2

/* System Libraries */
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <string>
#include <system_error>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
// #include "../include/libUtils.h"
#include "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch7/include/libUtils.h"

using namespace std;
using namespace cv;

/* ==================== */
/*  OpenCV's Functions  */
/* ==================== */
int orb_nfeatures = 100; //FIXME: move to main
double matches_lower_bound = 30.0;  //FIXME: move to main


typedef chrono::steady_clock::time_point Timer;  // FIXME: Remover, importar da libUtils.h
void printTimeElapsed(const char text[], Timer t1, Timer t2){ // FIXME: Remover, importar da libUtils.h
    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << text << time_elapsed.count() << " s" << endl << endl;
}

void find_features_matches(const Mat &image1, const Mat &image2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &goodMatches, bool verbose){
    //--- Initialization
    Mat descriptors1, descriptors2;

    #ifdef OPENCV3
        // cout << "'OpenCV3' selected." << endl << endl;
        Ptr<FeatureDetector> detector = ORB::create(orb_nfeatures);
        Ptr<DescriptorExtractor> descriptor = ORB::create(orb_nfeatures);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    #else
        // cout << "'OpenCV2' selected." << endl << endl;
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
    if(verbose){
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

    /* Display */
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);
    imshow("image_goodMatches", image_goodMatches);
}
