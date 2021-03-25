#ifndef bundleAdjustment_H_
#define bundleAdjustment_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "bundleAdjustment.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

#endif