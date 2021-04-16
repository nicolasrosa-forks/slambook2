/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <string>
#include <system_error>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../common/libUtils_basic.h"
#include "../../common/libUtils_eigen.h"
#include "../../common/libUtils_opencv.h"
#include "../include/OpticalFlowTracker.h"

using namespace std;
using namespace cv;

/* ============================== */
/*  Class Methods Implementation  */
/* ============================== */
// void OpticalFlowTracker::OpticalFlowTracker(){

// }

void OpticalFlowTracker::calculateOpticalFlow(const Range &range){
    cout << "heloo" << endl;
}