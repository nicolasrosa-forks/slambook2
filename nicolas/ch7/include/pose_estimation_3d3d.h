#ifndef pose_estimation_3d3d_ICP_H_
#define pose_estimation_3d3d_ICP_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "pose_estimation_3d3d.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void ICP_SVD(
    const vector<Point3f> &pts1_p,
    const vector<Point3f> &pts2_p,
    Mat &R, Mat &t
);

void ICP_bundleAdjustment(
    const vector<Point3f> &pts1_p,
    const vector<Point3f> &pts2_p,
    Mat &R, Mat &t
);

#endif