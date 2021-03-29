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

Point2f pixel2cam(const Point2f &p, const Mat &K);

// Point2f cam2pixel(const Point2f &x, const Mat &K); // TODO

Eigen::Vector2d cam2pixel(const Eigen::Vector3d &P, const Eigen::Matrix3d &K);

#endif