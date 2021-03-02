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

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv] Hello!" << endl;

    /* Load the images */
    Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    /* Step 1 - Calculate Oriented FAST keypoints */
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    
    /* Step 2: Calculate BRIEF descriptors based on the position of Oriented FAST keypoints */
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

    Mat outImage1, outImage2;
    drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    /* Step 3: Match BRIEF descriptors of the two images using Hamming distance */
    vector<DMatch> matches;
    
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);
    t2 = chrono::steady_clock::now();

    printTimeElapsed("ORB Features Matching: ", t1, t2);

    cout << "-- Number of matches: " << matches.size() << endl;

    /* Step 4: Select correct matching (filtering) */
    // Calculate the min & max distances
    
    /* Parameters: __first – Start of range.
                   __last – End of range.
                   __comp – Comparison functor.
    */
    auto min_max = minmax_element(matches.begin(), matches.end(), 
        [](const DMatch &m1, const DMatch &m2) {
        //cout << m1.distance << " " << m2.distance << endl;
        return m1.distance < m2.distance;});  // Return a pair of iterators pointing to the minimum and maximum elements in a range.
    
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Min dist: %f \n", min_dist);
    printf("-- Max dist: %f \n\n", max_dist);

    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching as wrong.
    // But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
    vector<DMatch> goodMatches;

    for (int i=0; i<descriptors1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
        }
    }

    cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;

    /* Step 5: Visualize the Matching result */
    Mat image_matches;
    Mat image_goodMatches;

    drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches);
    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches);

    /* Display */
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);  
    imshow("image_matches", image_matches);
    imshow("image_goodMatches", image_goodMatches); 
    waitKey(0);

    cout << "\nDone." << endl;

    return 0;
}
