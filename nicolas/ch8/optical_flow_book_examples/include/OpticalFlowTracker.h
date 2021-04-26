#ifndef OPTICALFLOWTRACKER_H_
#define OPTICALFLOWTRACKER_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
class OpticalFlowTracker{
public:
    // Constructor
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kps1_,
        vector<KeyPoint> &kps2_,
        vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kps1(kps1_), kps2(kps2_), success(success_), inverse(inverse_),
        has_initial_guess(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kps1;
    vector<KeyPoint> &kps2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial_guess = false;
};

#include "OpticalFlowTracker.cpp"

#endif