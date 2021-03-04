#ifndef LIBUTILS_H_
#define LIBUTILS_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "libUtils.cpp"

/* ================== */
/*  Functions Scopes  */
/* ================== */
void print(char text[]);

void printVec(const char text[], const std::vector<double> &vec);

template <typename TTypeMat>
void printMatrix(const char text[], TTypeMat mat);

template <typename TTypeVec>
void printVector(const char text[], TTypeVec vec);

template <typename TTypeQuat>
void printQuaternion(const char text[], TTypeQuat quat);

int checkImage(const cv::Mat &image);

void printImageInfo(const cv::Mat &image);

void printMatrix(const char text[], cv::Mat var);

#endif