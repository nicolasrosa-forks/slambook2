#ifndef pose_estimation_2d2d_H_
#define pose_estimation_2d2d_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "pose_estimation_2d2d.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void pose_estimation_2d2d(
    const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,
    const vector<DMatch> &matches,
    Mat &R, Mat &t, Mat &K);

#endif