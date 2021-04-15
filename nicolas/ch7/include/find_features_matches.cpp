/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <string>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// #ifdef HAVE_OPENCV_XFEATURES2D
// #include <opencv2/xfeatures2d/xfeatures2d.hpp>

/* Custom Libraries */
#include "../../common/libUtils_basic.h"
#include "../../common/libUtils_eigen.h"
#include "../../common/libUtils_opencv.h"

using namespace std;
using namespace cv;
// using namespace cv::xfeatures2d;

/** Features Descriptors
 * Binary-string descriptors: ORB, BRIEF, BRISK, FREAK, AKAZE, etc.
 * Floating-point descriptors: SIFT, SURF, GLOH, etc.
 * 
 * @ORB: Free, 
 *  - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
 * @SIFT: Paid, 
 *  - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
 *  - https://www.youtube.com/watch?v=mHWjdav8KVk
 * @SURF: Paid, # TODO
 *  - Feature Detection: https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html
 *  - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html

/** Non-Maximum Suppression
 * @PAPER: "Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution"
 * - https://github.com/BAILOOL/ANMS-Codes
 */

/** Features Matching
 * Feature matching of binary descriptors can be efficiently done by comparing their Hamming distance as opposed to 
 * Euclidean distance used for floating-point descriptors.
 * 
 * For comparing binary descriptors in OpenCV, use FLANN + LSH index or Brute Force + Hamming distance.
 * 
 * https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
 * 
 * @BRUTEFORCE: 
 *  - Hamming Distance: https://en.wikipedia.org/wiki/Hamming_distance
 * @FLANN:
 *  - Lowe's ratio test: https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94
 *  - https://docs.opencv.org/3.4/d5/dde/tutorial_feature_description.html
 *  - Feature Matching with FLANN: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
 */


/* Global Variables */
// Choose the Feature/Matcher Method:
const char* feature_matcher_enum2str[] = {
    "ORB + BruteForce (Hamming)", 
    "ORB + FlannBased", // FIXME: Can I use this?
    "SIFT + BruteForce (Norm L2)",
    "SIFT + FlannBased"
};
int feature_matcher_selected = 1;

// Matchers Params
double matches_lower_bound = 30.0;
const float ratio_threshold = 0.6f;

//
bool applyANMS = false;
bool isFirstTime = true;

/* =========== */
/*  Functions  */
/* =========== */
void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint> &keypoints, const int numToKeep){
    if( keypoints.size() < numToKeep ) { return; }

    //
    // Sort by response
    //
    std::sort( keypoints.begin(), keypoints.end(), [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs ){
        return lhs.response > rhs.response;
    });

    std::vector<cv::KeyPoint> anmsPts;

    std::vector<double> radii;
    radii.resize( keypoints.size() );
    std::vector<double> radiiSorted;
    radiiSorted.resize( keypoints.size() );

    const float robustCoeff = 1.11; // see paper

    for( int i = 0; i < keypoints.size(); ++i ){
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( int j = 0; j < i && keypoints[j].response > response; ++j ){
            radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
    }

    std::sort( radiiSorted.begin(), radiiSorted.end(), [&]( const double& lhs, const double& rhs ){
        return lhs > rhs;
    });

    const double decisionRadius = radiiSorted[numToKeep];
    for( int i = 0; i < radii.size(); ++i ){
        if( radii[i] >= decisionRadius ){
            anmsPts.push_back( keypoints[i] );
        }
    }

    anmsPts.swap( keypoints );
}

void find_features_matches(const Mat &image1, const Mat &image2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &goodMatches, int nfeatures, bool verbose){
    /* Initialization */
    Mat descriptors1, descriptors2;

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;
    Ptr<DescriptorMatcher> matcher;

    string descriptorType = "";
            
    switch (feature_matcher_selected){
        case 1:  // ORB + BruteForce-Hamming, Descriptor type: CV_8U
            detector = ORB::create(nfeatures);
            descriptor = ORB::create(nfeatures);
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
            descriptorType = "binary-string";
            break;
        case 2: // ORB + FlannBased, Descriptor type: CV_8U # FIXME: Makes sense? Apparently yes, but you need to use specified Search params.
            detector = ORB::create(nfeatures);
            descriptor = ORB::create(nfeatures);
            matcher = DescriptorMatcher::create("FlannBased");
            descriptorType = "binary-string";
            break;
        case 3: // SIFT + BruteForce, Descriptor type: CV_32F
            detector = SIFT::create(nfeatures);
            descriptor = SIFT::create(nfeatures);
            matcher = DescriptorMatcher::create("BruteForce");  // It uses Norm L2
            descriptorType = "float-point";
            break;
        case 4: // SIFT + FlannBased, Descriptor type: CV_32F
            detector = SIFT::create(nfeatures);
            descriptor = SIFT::create(nfeatures);
            matcher = DescriptorMatcher::create("FlannBased");
            descriptorType = "float-point";
            break;
        case 5: // SURF +
            detector = cv::xfeatures2d::SURF::create(nfeatures);
        default:
            break;
    }

    if(isFirstTime)
        cout << "-- Feature/Matcher selected: " << feature_matcher_enum2str[feature_matcher_selected-1] << endl;

    /* --------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    //--- Step 1: Detect the position of the keypoints (Corner Points)
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    Timer t2 = chrono::steady_clock::now();

    // if(applyANMS){
    //     adaptiveNonMaximalSuppresion(keypoints1, 500);
    //     adaptiveNonMaximalSuppresion(keypoints2, 500);
    // }
    
    //--- Step 2: Calculate the descriptors based on the position of keypoints
    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);
    Timer t3 = chrono::steady_clock::now();

    //cout << descriptors1 << endl;
    //cout << descriptors2 << endl;

    // cout << "descriptors1: " << descriptors1.type() << endl;
    // cout << "descriptors2: " << descriptors2.type() << endl;
    // cin.ignore();

    /* ------------------- */
    /*  Features Matching  */
    /* ------------------- */
    //--- Step 3: Match the descriptors of the two images using the appropriate matching metric (Hamming or Euclidean distances)
    vector<DMatch> matches;
    vector<vector<DMatch>> knn_matches;

    Timer t4 = chrono::steady_clock::now();
    if(descriptorType.compare("binary-string") == 0){
        matcher->match(descriptors1, descriptors2, matches);
    }else if(descriptorType.compare("float-point") == 0){
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    }
    Timer t5 = chrono::steady_clock::now();

    /* -------------------- */
    /*  Features Filtering  */
    /* -------------------- */
    //--- Step 4: Correct matching selection
    Timer t6, t7;
    double min_dist = 10000, max_dist = 0;
    int n_matches = 0;

    if(descriptorType.compare("binary-string") == 0){  // Hamming Distance
        /* Calculate the min & max distances */
        // Find the minimum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
        t6 = chrono::steady_clock::now();
        for (int i = 0; i < descriptors1.rows; i++){
            double dist = matches[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;
        }

        /* Perform Filtering */
        // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching
        // as wrong. But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
        // vector<DMatch> goodMatches;

        for (int i=0; i<descriptors1.rows; i++){
            // cout << matches[i].distance << endl;
            if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
                goodMatches.push_back(matches[i]);
            }
        }
        
        n_matches = matches.size();
        t7 = chrono::steady_clock::now();
    
    }else if(descriptorType.compare("float-point") == 0){  // Lowe's ratio test
        // Filter matches using the Lowe's ratio test        
        t6 = chrono::steady_clock::now();
        for(size_t i=0; i < knn_matches.size(); i++){
            if(knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance){
                goodMatches.push_back(knn_matches[i][0]);
            }
        }
        
        n_matches = knn_matches.size();
        t7 = chrono::steady_clock::now();
    }
    
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
    if(verbose){
        printElapsedTime("Features Extraction: ", t1, t3);
        printElapsedTime(" | Keypoints detection: ", t1, t2);
        printElapsedTime(" | Descriptors calculation: ", t2, t3);
        cout << "\n-- Number of detected keypoints1: " << keypoints1.size() << endl;
        cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

        printElapsedTime("Features Matching: ", t4, t5);
        cout << "-- Number of matches: " << matches.size() << endl << endl;

        printElapsedTime("Features Filtering: ", t6, t7);
        if(descriptorType.compare("binary-string") == 0){  // Hamming Distance
            printElapsedTime(" | Filtering by Hamming Distance: ", t6, t7);
            cout << "-- Min dist: " << min_dist << endl;
            cout << "-- Max dist: " << max_dist << endl;
        }else if(descriptorType.compare("float-point") == 0){  // Lowe's ratio test
            printElapsedTime(" | Filtering by Lowe's ratio test: ", t6, t7);
        }
        cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;
    }

    cout << "In total, we get " << goodMatches.size() << "/" << n_matches << " good pairs of feature points." << endl << endl;
    // cout << "\r" << "In total, we get " << goodMatches.size() << "/" << n_matches << " good pairs of feature points." << std::flush;  //TODO:

    /* Display */
    imshow("outImage1", outImage1);
    imshow("outImage2", outImage2);
    imshow("image_matches", image_matches);
    imshow("image_goodMatches", image_goodMatches);

    isFirstTime = false;
}