/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <exception>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"
#include "../../../common/libUtils_fps.h"
#include "../../../common/connected_components.h"
#include "../include/optical_flow.h"

using namespace std;
using namespace cv;

/* Global Variables */
// Choose:
string filename = "/home/nicolas/Downloads/Driving_Downtown_-_New_York_City_4K_-_USA_360p.mp4";
// string filename = "/home/nicolas/Downloads/Driving_Downtown_-_San_Francisco_4K_-_USA_720p.mp4";

int nfeatures = 500;
int min_nfeatures = 250;

Mat calculateDenseFlow(Mat prvs, Mat next){
    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    
    return bgr;
}

const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
const String window_event_name = "Event";
int low_Diff = 0, high_Diff = max_value;
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

static void on_diff_thresh_trackbar(int, void *)
{
    low_Diff = min(high_Diff, low_Diff+1);
    setTrackbarPos("Diff Thresh", window_event_name, low_Diff);
}


// Glare Detection
// https://rcvaram.medium.com/glare-removal-with-inpainting-opencv-python-95355aa2aa52

// def create_mask(image):
//     gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
//     blurred = cv2.GaussianBlur( gray, (9,9), 0 )
//     _,thresh_img = cv2.threshold( blurred, 180, 255, cv2.THRESH_BINARY)
//     thresh_img = cv2.erode( thresh_img, None, iterations=2 )
//     thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
// 
//     # perform a connected component analysis on the thresholded image,
//     # then initialize a mask to store only the "large" components
//     labels = measure.label( thresh_img, neighbors=8, background=0 )
//     mask = np.zeros( thresh_img.shape, dtype="uint8" )
// 
//     # loop over the unique components
//     for label in np.unique( labels ):
//         # if this is the background label, ignore it
//         if label == 0:
//             continue
//
//         # otherwise, construct the label mask and count the
//         # number of pixels
//         labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
//         labelMask[labels == label] = 255
//         numPixels = cv2.countNonZero( labelMask )
//
//         # if the number of pixels in the component is sufficiently
//         # large, then add it to our mask of "large blobs"
//         if numPixels > 300:
//             mask = cv2.add( mask, labelMask )
//     return mask

Mat create_glare_mask(Mat image){
    Mat gray, blurred, thresh_img;  // FIXME: gray -> grey
    cv::cvtColor(image, gray, COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0);
    cv::threshold( blurred, thresh_img, 180, 255, THRESH_BINARY);
    cv::erode(thresh_img, thresh_img, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1,-1), 2);
    cv::dilate(thresh_img, thresh_img, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1,-1), 4);

    // Perform a connected component analysis on the thresholded image,
    // then initialize a mask to store only the "large" components
    // cv::measure.label
    std::vector<ConnectedComponent> labels;

    findCC(thresh_img, labels);

    int n_labels = labels.size();

    Mat mask = cv::Mat::zeros(thresh_img.size(), CV_8UC1);

    for(size_t i=0; i<n_labels; i++){
        // if this is the background label, ignore it
        if(i==0)
            continue;

        // otherwise, construct the label mask and count the
        // number of pixels
        Mat labelMask = cv::Mat::zeros(thresh_img.size(), CV_8UC1);
        
        for(auto px: (*labels[i].getPixels())){
            // cout << px << endl;
            labelMask.at<uchar>(px) = 255;
        }
        
        int numPixels = cv::countNonZero(labelMask);

        // imshow("labelMask", labelMask);
        cout << "label[" << i << "], numPixels: " << numPixels << endl;

        // if the number of pixels in the component is sufficiently
        // large, then add it to our mask of "large blobs"
        if(numPixels > 300){
            cv::add(mask, labelMask, mask);
        }
    }

    float alpha = 0.5;
    float beta = ( 1.0 - alpha);
    Mat glare_overlay = cv::Mat::zeros(image.size(), CV_8UC1);
    addWeighted( gray, alpha, mask, beta, 0.0, glare_overlay);

    // cout << image.size() << "vs." << mask.size() << endl;

    imshow("grey", gray);
    imshow("blurred", blurred);
    imshow("thresh_img", thresh_img);
    imshow("mask", mask);
    imshow("glare_overlay", glare_overlay);
    
    return mask;
}

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) { // FIXME: Acho que não está funcionando corretamente.
    cout << "[orb_cv_video] Hello!" << endl << endl;

    /* Load the images */
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap(filename);  // Create a VideoCapture object and open the input file

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Get Original FPS from the video
    double fps_v = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps_v << endl;

    /* Initialization */
    Mat image1_bgr, image2_bgr;
    Mat image1, image2;
    vector<KeyPoint> kps1;
    vector<KeyPoint> kps2;
    vector<Point2f> pts1_2d;
    vector<Point2f> pts2_2d;
    
    // Optical Flow Variables
    Ptr<GFTTDetector> detector = GFTTDetector::create(nfeatures, 0.01, 20);
    
    // vector<KeyPoint> cv_flow_kps2;    // Estimated KeyPoints in Image 2 by Multi-Level Optical Flow
    vector<Point2f> cv_flow_pts1_2d;  // Coordinates of Tracked Keypoints in Image 1
    vector<Point2f> cv_flow_pts2_2d;  // Coordinates of Tracked Keypoints in Image 2
    vector<uchar> cv_flow_status12;
    vector<uchar> cv_flow_status21;
    vector<float> cv_flow_error12;
    vector<float> cv_flow_error21;
    
    Mat cv_flow_outImage12;
    Mat cv_flow_outImage12_diffImage;
    Mat cv_flow_outImage12_blackImage;
    Mat cv_flow_outImage12_eventImage;


    Mat cv_flow_outImage21;

    // Get first frame
    cap >> image1_bgr;
    assert(image1_bgr.data != nullptr);  // FIXME: I think this its not working!

    cv::cvtColor(image1_bgr, image1, COLOR_BGR2GRAY);
    detector->detect(image1, kps1);
    for(auto &kp: kps1) pts1_2d.push_back(kp.pt);

    // Variables for FPS Calculation
    FPS fps = FPS();
    
    // Sharpen Filter // TODO: Manter?
    // Mat blurred; double sigma = 1, threshold = 5, amount = 1;
    
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    namedWindow(window_event_name);

    // Trackbars to set thresholds for HSV values
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    createTrackbar("Diff Threshold", window_event_name, &low_Diff, high_Diff, on_low_H_thresh_trackbar);
    
    Mat frame, frame_HSV, frame_threshold;

    /* ------ */
    /*  Loop  */
    /* ------ */
    while(1){
        /* Read */
        // Capture frame-by-frame
        cap >> image2_bgr;

        // If the frame is empty, break immediately
        if (image2_bgr.empty())
            break;

        // cout << "Width : " << image2_bgr.size().width << endl;
        // cout << "Height: " << image2_bgr.size().height << endl;

        /* Apply Filters */ // TODO: Manter?
        // GaussianBlur(image2_bgr, blurred, Size(), sigma, sigma);
        // Mat lowContrastMask = abs(image2_bgr - blurred) < threshold;
        // Mat sharpened = image2_bgr*(1+amount) + blurred*(-amount);
        // image2_bgr.copyTo(sharpened, lowContrastMask);
        // sharpened.copyTo(image2_bgr); // Overwrite
        // imshow("sharpened", sharpened);
        // imshow("lowContrastMask", lowContrastMask);

        /* ----- Features Extraction and Matching ----- */
        cv::cvtColor(image2_bgr, image2, COLOR_BGR2GRAY);
        
        /* ----- Apply Histogram Equalization ----- */
        // Mat image2_grey_est;
        // calcHist(image2_grey, "image2_grey");  // Before Equalization
        // equalizeHist( image2_grey, image2_grey_est );  // Input needs to be greyscale!
        // calcHist(image2_grey_est, "image2_grey_est");  // After Equalization
        // image2_grey_est.copyTo(image2_grey); // Overwrite

        /* ----- Optical Flow ----- */
        cv::calcOpticalFlowPyrLK(image1, image2, pts1_2d, cv_flow_pts2_2d, cv_flow_status12, cv_flow_error12);  // Fills the pts2_2d with the corresponding keypoints tracked in Image 2.

        Mat denseFlow12 = calculateDenseFlow(image1, image2);
        Mat denseFlow21 = calculateDenseFlow(image2, image1);

        Mat denseFlow12_grey, denseFlow21_grey;
        cv::cvtColor(denseFlow12, denseFlow12_grey, COLOR_BGR2GRAY);
        cv::cvtColor(denseFlow21, denseFlow21_grey, COLOR_BGR2GRAY);

        // FIXME: Bi-directional Optical Flow Consistency (Frame2 -> Frame1)
        // detector->detect(image2, kps2);
        // for(auto &kp: kps2) pts2_2d.push_back(kp.pt);
        if(pts2_2d.size()>0)
            // detector->detect(image1, kps1);
            // for(auto &kp: kps1) pts1_2d.push_back(kp.pt);
            cv::calcOpticalFlowPyrLK(image2, image1, pts2_2d, cv_flow_pts1_2d, cv_flow_status21, cv_flow_error21);  // Fills the pts2_2d with the corresponding keypoints tracked in Image 2.

        // Diff Image
        cv::Mat diffImage;
        
        cv::absdiff(image1, image2, diffImage);
        // diffImage2 = image2-image1;
        
        cv::Mat eventImage(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));

        // for (int v=0; v<image2.rows; v++){
        //     for (int u=0; u<image2.cols; u++){
        //         // if(diffImage2.ptr<unsigned short>(v)[u] != 0)
        //         if(diffImage.at<uchar>(v, u) > 100){
        //             cout << diffImage.at<uchar>(v, u) << endl;
        //             // cout << diffImage.channels() << diffImage.type() << endl;

        //             if(image1.ptr<uchar>(v)[u] < image2.ptr<uchar>(v)[u])
        //                 event.at<Vec3b>(v,u)[2] = 250;  // Red
        //             else
        //                 event.at<Vec3b>(v,u)[0] = 250;  // Blue
        //         }
        //     }
        // }

        parallel_for_(Range(0, image2.rows*image2.cols), [&](const Range& range){
            for (int r = range.start; r < range.end; r++){
                int v = r / image2.cols;
                int u = r % image2.cols;
                
                if(diffImage.at<uchar>(v, u) > low_Diff){
                // if(diffImage.at<uchar>(v, u) > 85){
                    // cout << diffImage.at<uchar>(v, u) << endl;
                    // cout << diffImage.channels() << diffImage.type() << endl;

                    if(image1.ptr<uchar>(v)[u] < image2.ptr<uchar>(v)[u])
                        eventImage.at<Vec3b>(v,u)[2] = 250;  // Red
                    else
                        eventImage.at<Vec3b>(v,u)[0] = 250;  // Blue
                }
            }
        });

        cv::Mat cv_flow_outImage12_eventImage = eventImage.clone();


        // cout << diffImage.size() << endl;

        Mat mask = create_glare_mask(image2_bgr);

        /* ----- Results ----- */
        drawOpticalFlow<uchar>(image2, cv_flow_outImage12, pts1_2d, cv_flow_pts2_2d, cv_flow_status12);
        drawOpticalFlow<uchar>(diffImage, cv_flow_outImage12_diffImage, pts1_2d, cv_flow_pts2_2d, cv_flow_status12);
        
        drawOpticalFlow<uchar>(image1, cv_flow_outImage21, pts2_2d, cv_flow_pts1_2d, cv_flow_status21);

        cv::Mat black(360, 640, CV_8UC1, cv::Scalar(0, 0, 0));
        
        drawOpticalFlow<uchar>(black, cv_flow_outImage12_blackImage, pts1_2d, cv_flow_pts2_2d, cv_flow_status12);
        drawOpticalFlow_red<uchar>(cv_flow_outImage12_blackImage, cv_flow_outImage12_blackImage, pts2_2d, cv_flow_pts1_2d, cv_flow_status21);
        drawOpticalFlow<uchar>(eventImage, cv_flow_outImage12_eventImage, pts1_2d, cv_flow_pts2_2d, cv_flow_status12);

        // drawOpticalFlow_red<uchar>(event, cv_flow_outImage12_event, pts2_2d, cv_flow_pts1_2d, cv_flow_status21);

        vector<Point2f> good_pts2_2d;
        for(uint i = 0; i < pts1_2d.size(); i++){
            // Select good points
            if(cv_flow_status12[i] == 1) {
                good_pts2_2d.push_back(cv_flow_pts2_2d[i]);
            }
        }
        int n_good21 = good_pts2_2d.size();
        cout << n_good21 << "/" << nfeatures << endl;

        vector<Point2f> good_pts1_2d;
        for(uint i = 0; i < pts2_2d.size(); i++){
            // Select good points
            if(cv_flow_status21[i] == 1) {
                good_pts1_2d.push_back(cv_flow_pts1_2d[i]);
            }
        }
        int n_good12 = good_pts1_2d.size();
        cout << n_good12 << "/" << nfeatures << endl;

        // Convert from BGR to HSV colorspace
        frame = image2_bgr.clone();
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        // Mat hsv;
        vector<Mat> channels;
        split(frame_HSV, channels);
        // Detect the object based on HSV Range Values
        inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
        // Show the frames
        imshow(window_capture_name, frame);
        imshow(window_detection_name, frame_threshold);

        // Display
        // imshow("Frame1", image1);
        // imshow("Frame2", image2);
        imshow("diffImage", diffImage);
        imshow("image2_bgr", image2_bgr);
        // imshow("image2_grey", image2_grey);
        imshow("Tracked by OpenCV (1->2)", cv_flow_outImage12);
        imshow("Tracked by OpenCV (2->1)", cv_flow_outImage21);
        imshow("Tracked by OpenCV (diffImage)", cv_flow_outImage12_diffImage);
        // imshow("diffImage2", diffImage2);
        imshow("Tracked by OpenCV (black)", cv_flow_outImage12_blackImage);
        imshow("Event", eventImage);
        imshow("Tracked by OpenCV (eventImage)", cv_flow_outImage12_eventImage);
        imshow("H", channels[0]);
        imshow("S", channels[1]);
        imshow("V", channels[2]);
        imshow("denseFlow (1->2)", denseFlow12);
        imshow("denseFlow (2->1)", denseFlow21);
        imshow("denseFlow (1->2)_grey", denseFlow12_grey);
        imshow("denseFlow (2->1)_grey", denseFlow21_grey);


        Mat denseFlow_consistency;
        bitwise_and(denseFlow12_grey,denseFlow21_grey,denseFlow_consistency);
        imshow("denseFlow (Consistency)",denseFlow_consistency);

        Mat denseFlow_consistency_bgr;
        cv::applyColorMap(denseFlow_consistency, denseFlow_consistency_bgr, 2);  // ColorMap: Jet
        imshow("denseFlow (Consistency, BGR)",denseFlow_consistency_bgr);


        /* ----- End Iteration ----- */
        // Next Iteration Prep
        image1 = image2.clone();  // Save last frame
        
        if (n_good12 < min_nfeatures){  // Few Features, get detect more!
            detector->detect(image2, kps2);
            for(auto &kp: kps2) pts2_2d.push_back(kp.pt);
            for(auto &pt: good_pts1_2d) pts2_2d.push_back(pt); // Retains previously detected keypoints
        }else{
            pts2_2d = good_pts1_2d;
        }

        if (n_good21 < min_nfeatures){  // Few Features, get detect more!
        // if (true){  // Few Features, get detect more!
            detector->detect(image1, kps1);
            for(auto &kp: kps1) pts1_2d.push_back(kp.pt);
            for(auto &pt: good_pts2_2d) pts1_2d.push_back(pt); // Retains previously detected keypoints
        }else{
            pts1_2d = good_pts2_2d;
        }

        // Free vectors
        // cv_flow_kps2.clear();
        cv_flow_pts2_2d.clear();
        cv_flow_status12.clear();
        
        // cv_flow_kps1.clear();
        cv_flow_pts1_2d.clear();
        cv_flow_status21.clear();
        

        // FPS Calculation
        fps.update();

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