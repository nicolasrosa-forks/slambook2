/** Features
 * @ORB: https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html
 * @SURF: https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html
 * 
 */

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
#include "../../common/libUtils_basic.h"
#include "../../common/libUtils_eigen.h"
#include "../../common/libUtils_opencv.h"

using namespace std;
using namespace cv;

/* ==================== */
/*  OpenCV's Functions  */
/* ==================== */
double matches_lower_bound = 30.0;

void find_features_matches(const Mat &image1, const Mat &image2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &goodMatches, int nfeatures, bool verbose){
    //--- Initialization
    Mat descriptors1, descriptors2;

    #ifdef OPENCV3
//        cout << "'OpenCV3' selected." << endl << endl;
        // Ptr<FeatureDetector> detector = SIFT::create(500); // TODO: Terminar
        // Ptr<DescriptorExtractor> descriptor = SIFT::create(500);
        Ptr<FeatureDetector> detector = ORB::create(nfeatures);
        Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    #else
//        cout << "'OpenCV2' selected." << endl << endl;
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

    //--- Step 3: Match the BRIEF descriptors of the two images using the Hamming distance
    vector<DMatch> matches;

    // if(descriptors1.type()!=CV_32F) {
    //     descriptors1.convertTo(descriptors1, CV_32F);
    // }

    // if(descriptors2.type()!=CV_32F) {
    //     descriptors2.convertTo(descriptors2, CV_32F);
    // }

    Timer t4 = chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);  // TODO: Ver como rodar aquele algoritmo de matching FLANN (Parece ser melhor quando tem-se muitos pontos)
    Timer t5 = chrono::steady_clock::now();

    //--- Step 4: Select correct matching (filtering)
    // Calculate the min & max distances
    double min_dist = 10000, max_dist = 0;

    // Find the minimum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
    Timer t6 = chrono::steady_clock::now();
    for (int i = 0; i < descriptors1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching as wrong.
    // But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
//    vector<DMatch> goodMatches;

    Timer t7 = chrono::steady_clock::now();
    for (int i=0; i<descriptors1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
        }
    }
    Timer t8 = chrono::steady_clock::now();

    //--- Step 5: Visualize the Matching result
    Mat outImage1, outImage2;
    Mat image_matches;
    Mat image_goodMatches;

    drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches);
    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches);

    /* Results */
    if(verbose){
        printElapsedTime("ORB Features Extraction: ", t1, t3);
        printElapsedTime(" | Oriented FAST Keypoints detection: ", t1, t2);
        printElapsedTime(" | BRIEF descriptors calculation: ", t2, t3);
        cout << "-- Number of detected keypoints1: " << keypoints1.size() << endl;
        cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

        printElapsedTime("ORB Features Matching: ", t4, t5);
        cout << "-- Number of matches: " << matches.size() << endl << endl;
        
        printElapsedTime("ORB Features Filtering: ", t6, t8);
        printElapsedTime(" | Min & Max Distances Calculation: ", t6, t7);
        printElapsedTime(" | Filtering by Hamming Distance: ", t7, t8);
        cout << "-- Min dist: " << min_dist << endl;
        cout << "-- Max dist: " << max_dist << endl;
        cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;
    }

    cout << "In total, we get " << goodMatches.size() << "/" << matches.size() << " good pairs of feature points." << endl << endl;

    /* Display */
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);
    imshow("image_matches", image_matches);
    imshow("image_goodMatches", image_goodMatches);
}
