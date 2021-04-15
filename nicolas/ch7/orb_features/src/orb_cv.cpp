/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../../orb_features/src/1.png";
string image2_filepath = "../../orb_features/src/2.png";

int nfeatures = 500;
double matches_lower_bound = 30.0;

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv] Hello!" << endl << endl;

    /* Load the images */
    Mat image1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    /* --------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    Ptr<FeatureDetector> detector = ORB::create(nfeatures);
    Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

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


    /* ------------------- */
    /*  Features Matching  */
    /* ------------------- */
    //--- Step 3: Match the BRIEF descriptors of the two images using the Hamming distance
    vector<DMatch> matches;

    Timer t4 = chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);
    Timer t5 = chrono::steady_clock::now();

    /* -------------------- */
    /*  Features Filtering  */
    /* -------------------- */
    //--- Step 4: Correct matching selection
    /* Calculate the min & max distances */
    /** Parameters: 
     * @param[in] __first – Start of range.
    /* @param[in] __last – End of range.
    /* @param[in] __comp – Comparison functor.
    /* @param[out] make_pair(m,M) Return a pair of iterators pointing to the minimum and maximum elements in a range.
     */
    Timer t6 = chrono::steady_clock::now();
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
        //cout << m1.distance << " " << m2.distance << endl;
        return m1.distance < m2.distance;
    });  

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    /* Perform Filtering */
    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching
    // as wrong. But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
    vector<DMatch> goodMatches;

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

    drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    /* Results */
    printElapsedTime("ORB Features Extraction: ", t1, t3);
    printElapsedTime(" | Oriented FAST Keypoints detection: ", t1, t2);
    printElapsedTime(" | BRIEF descriptors calculation: ", t2, t3);
    cout << "\n-- Number of detected keypoints1: " << keypoints1.size() << endl;
    cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

    printElapsedTime("ORB Features Matching: ", t4, t5);
    cout << "-- Number of matches: " << matches.size() << endl << endl;
    
    printElapsedTime("ORB Features Filtering: ", t6, t8);
    printElapsedTime(" | Min & Max Distances Calculation: ", t6, t7);
    printElapsedTime(" | Filtering by Hamming Distance: ", t7, t8);
    cout << "-- Min dist: " << min_dist << endl;
    cout << "-- Max dist: " << max_dist << endl;
    cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;
    cout << "In total, we get " << goodMatches.size() << "/" << matches.size() << " good pairs of feature points." << endl << endl;

    /* Display */
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);
    imshow("image_matches", image_matches);
    imshow("image_goodMatches", image_goodMatches);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    /* Save */
    imwrite("../../orb_features/src/results_orb_cv_goodMatches.png", image_goodMatches);

    cout << "Done." << endl;

    return 0;
}