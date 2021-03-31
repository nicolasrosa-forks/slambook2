#ifndef pose_estimation_3d3d_ICP_H_
#define pose_estimation_3d3d_ICP_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "pose_estimation_3d3d_ICP.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void pose_estimation_3d3d(
    const vector<Point3f> &pts1,
    const vector<Point3f> &pts2,
    Mat &R, Mat &t
);

void bundleAdjustment(
    const vector<Point3f> &points_3d,
    const vector<Point3f> &points_2d, //FIXME: Isto está correto?
    Mat &R, Mat &t
);
#endif