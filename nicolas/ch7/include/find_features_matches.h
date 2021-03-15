#ifndef find_features_matches_H_
#define find_features_matches_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "find_features_matches.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void find_features_matches(
    const Mat &image1, const Mat &image2,
    vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
    vector<DMatch> &goodMatches,
    int orb_nfeatures,
    bool verbose);

#endif