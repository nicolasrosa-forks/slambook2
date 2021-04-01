/* Custom Libraries */
#include "../include/imageBasics.h"

using namespace std;

/* Global Variables */
string image_filepath = "../../imageBasics/src/dog.jpg";

/* ====== */
/*  Main  */
/* ====== */
/* This Program demonstrates the following operations: image reading, displaying, pixel vising, copying, assignment, etc */
int main(int argc, char **argv){
    print("[imageBasics] Hello!\n");

    // Read the image as 8UC3 (BGR)
    cout << "[imageBasics] Reading '" << image_filepath << "'...";
    cv::Mat image = cv::imread(image_filepath);  // call cv::imread() to read the image from file

    if(!checkImage(image)){
        return 0;
    }

    // Print some basic information
    printImageInfo("image", image);

    cv::imshow("image", image);  // Use cv::imshow() to show the image
    cv::waitKey(0);              // Display and wait for a keyboard input

    // Accessing Image Pixel data
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t y=0; y < image.rows; y++){  // [0, H-1]
        // Use cv::Mat::ptr to get the pointer of each row
        unsigned char *row_ptr = image.ptr<unsigned char>(y);  // row_ptr is the pointer to y-th row

        for(size_t x=0; x < image.cols; x++){  // [0, W-1]
            // Read the pixel on (x,y), x=column, y=row
            unsigned char *data_ptr = &row_ptr[x*image.channels()];  // data_ptr is the pointer to (x,y)

            for(size_t c=0; c!= image.channels();c++){
                unsigned char data = data_ptr[c];  // data should be pixel of I(x,y) in c-th channel
                // cout << "I(" << x << "," << y << "," << c << "): " << int(data) << endl;
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>> (t2-t1);
    cout << "time elapsed: " << time_elapsed.count() << " s" << endl;

    /* ----- Copying cv::Mat ----- */
    // Example 1
    // operator '=' will not copy the image data, but only the reference!
    cv::Mat image_modified = image;

    // Changing 'image_modified' will also change image
    image_modified(cv::Rect(0, 0, 100, 100)).setTo(0);  // Set top-left 100x100 block to 0 (Black)

    cv::imshow("image", image);
    cv::imshow("image_modified", image_modified);
    cv::waitKey(0);
    cv::destroyWindow("image_modified");

    // Example 2
    // Use cv::Mat::clone() to actually clone the data
    // cv::Mat image_clone = image;          // 'image' will be modified as the 'image_clone' (both with white rectangle)
    cv::Mat image_clone = image.clone();  // 'image' will not be modified ('image' with a black rectangle and 'image_clone' with white rectangle)

    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);  // Set top-left 100x100 block to 255 (White)

    cv::imshow("image", image);
    cv::imshow("image_modified2", image_clone);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "\nDone." << endl;
    return 0;
}
