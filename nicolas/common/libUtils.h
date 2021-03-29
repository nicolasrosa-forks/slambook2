#ifndef LIBUTILS_H_
#define LIBUTILS_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "libUtils.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
/* Basic */
void print(char text[]);

void print(const std::string &text);

void print(double var);

void printVec(const char text[], const std::vector<double> &vec);

template <typename TTypeVec>
TTypeVec slicing(TTypeVec &arr, int begin_idx, int end_idx);

/* Chrono */
void printElapsedTime(const char text[], Timer t1, Timer t2);

/* Eigen3/Sophus */
template <typename TTypeEigenMat>
void printMatrix(const char text[], TTypeEigenMat mat);

template <typename TTypeEigenVec>
void printVector(const char text[], TTypeEigenVec vec);

template <typename TTypeEigenQuat>
void printQuaternion(const char text[], TTypeEigenQuat quat);

/* OpenCV */
int checkImage(const cv::Mat &image);

string type2str(int type);

void printImageInfo(const cv::Mat &image);

void printMatrix(const char text[], cv::Mat var);

void printMatrix(const char text[], cv::MatExpr var);

// Pixel coordinates to camera normalized coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

#endif