#ifndef LIBUTILS_OPENCV_H_
#define LIBUTILS_OPENCV_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "libUtils_opencv.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
/* OpenCV */
int checkImage(const cv::Mat &image);

string type2str(int type);

void printImageInfo(const cv::Mat &image);

void printMatrix(const char text[], cv::Mat var);

void printMatrix(const char text[], cv::MatExpr var);

Point2f pixel2cam(const Point2f &p, const Mat &K);

Point2d pixel2cam2(const Point2d &p, const Mat &K);

// Point2f cam2pixel(const Point2f &x, const Mat &K); // TODO


#endif