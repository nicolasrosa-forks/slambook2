#ifndef OPTICAL_FLOW_H_
#define OPTICAL_FLOW_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "optical_flow.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kps1,
    vector<KeyPoint> &kps2,
    vector<bool> &success,
    bool inverse,
    bool has_initial_guess
);

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse
);

template <typename TType>
void drawOpticalFlow(
    const Mat &inImage,
    Mat &outImage,
    const vector<Point2f> &pts1_2d,
    const vector<Point2f> &pts2_2d,
    vector<TType> &status
);

#endif