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

/* ====== */
/*  Main  */
/* ====== */
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
    
    cout << "Number of detected keypoints1: " << keypoints1.size() << endl;
    cout << "Number of detected keypoints2: " << keypoints2.size() << endl << endl;
    
    /* Step 2: Calculate BRIEF descriptors based on the position of Oriented FAST keypoints */
    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);

    //cout << descriptors1 << endl;
    //cout << descriptors2 << endl;    
     
    Timer t2 = chrono::steady_clock::now();
    printTimeElapsed("ORB Features Extraction: ", t1, t2);

    Mat outImage1, outImage2;
    drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // Step 3: Match BRIEF descriptors of the two images using Hamming distance
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);
    t2 = chrono::steady_clock::now();
    printTimeElapsed("ORB Features Matching: ", t1, t2);
    
    /* Display */
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);  
    waitKey(0);

    cout << "\nDone." << endl;

    return 0;
}
