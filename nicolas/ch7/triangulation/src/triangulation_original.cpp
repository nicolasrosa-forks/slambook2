#include <iostream>
#include <opencv2/opencv.hpp>
// #include "extra.h" // used in opencv2
using namespace std;
using namespace cv;

string image1_filepath = "../../images/1.png";
string image2_filepath = "../../images/2.png";

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

void pose_estimation_2d2d(
    const std::vector<KeyPoint> &keypoints_1,
    const std::vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    Mat &R, Mat &t);

void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points
);

/// For drawing
inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th-low_th;
    if (depth> up_th) depth = up_th;
    if (depth <low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1-depth / th_range));
}

// Pixel coordinates to camera normalized coordinates
Point2f pixel2cam(const Point2f &p, const Mat &K);

int main(int argc, char **argv) {
    // if (argc != 3) {
    //   cout << "usage: triangulation img1 img2" << endl;
    //   return 1;
    // }

    //--- read image
    //Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    //Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    Mat img_1 = imread(image1_filepath, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(image2_filepath, CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "\nA total of found " << matches.size() << " group matching points." << endl;

    //--- estimate the motion between two images
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //--- triangulation
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    //--- verify the reprojection relationship between triangulation points and feature points
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    for (int i = 0; i <matches.size(); i++) {
        // first picture
        float depth1 = points[i].z;
        cout << "depth: "<< depth1 << endl;
        Point2f pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // second picture
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                                                    std::vector<KeyPoint> &keypoints_1,
                                                    std::vector<KeyPoint> &keypoints_2,
                                                    std::vector<DMatch> &matches) {
    //--- initialization
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ("ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ("ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //--- Step 1: Detect the position of the Oriented FAST corner point
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //--- Step 2: Calculate the BRIEF descriptor according to the corner position
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //--- Step 3: Match the BRIEF descriptors in the two images, using Hamming distance
    vector<DMatch> match;
    // BFMatcher matcher (NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //--- The fourth step: matching points to filter
    double min_dist = 10000, max_dist = 0;

    //Find the minimum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
    for (int i = 0; i <descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist <min_dist) min_dist = dist;
        if (dist> max_dist) max_dist = dist;
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    //When the distance between the descriptors is greater than twice the minimum distance, the match is considered wrong. But sometimes the minimum distance will be very small, set an experience value of 30 as the lower limit.
    for (int i = 0; i <descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimation_2d2d(
    const std::vector<KeyPoint> &keypoints_1,
    const std::vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    Mat &R, Mat &t) {
    // Camera internal parameters, TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //--- convert the matching point to the form of vector<Point2f>
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i <(int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //--- calculate the essential matrix
    Point2d principal_point(325.1, 249.7); //Camera principal point, TUM dataset calibration value
    int focal_length = 521; //Camera focal length, TUM dataset calibration value
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

    //--- restore rotation and translation information from the essential matrix.
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points) {
    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;
    for (DMatch m:matches) {
        // Convert pixel coordinates to camera coordinates
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // Convert to non-homogeneous coordinates
    for (int i = 0; i <pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // normalization
        Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0)
        );
        points.push_back(p);
    }
}

Point2f pixel2cam(const Point2f &p, const Mat &K) {
    return Point2f
        (
            (p.x-K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y-K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
}