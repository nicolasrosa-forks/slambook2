/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"

using namespace std;
using namespace cv;

/* Global Variables */
// string image1_filepath = "../../images/1.png";
// string image2_filepath = "../../images/2.png";


/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to use 2D-2D feature matching to estimate camera motion. */
int main(int argc, char **argv) {
    cout << "[direct_method] Hello!" << endl << endl;

    /* Load the images */

    /* Initialization */


    /* --------- */
    /*  Results  */
    /* --------  */
    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}