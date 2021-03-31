#ifndef pose_estimation_3d2d_bundleAdjustment_H_
#define pose_estimation_3d2d_bundleAdjustment_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "pose_estimation_3d2d_bundleAdjustment.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Eigen::Matrix3d &K,
    Sophus::SE3d &pose
);

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Eigen::Matrix3d &K,
    Sophus::SE3d &pose
);

#endif