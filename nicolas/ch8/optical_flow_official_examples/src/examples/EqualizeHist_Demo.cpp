/**
 * @function EqualizeHist_Demo.cpp
 * @brief Demo code for equalizeHist function
 * @author OpenCV team
 * @link https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //! [Load image]
    CommandLineParser parser( argc, argv, "{@input | lena.jpg | input image}" );
    // Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
    char filename[] = "../../optical_flow_official_examples/src/examples/low-contrast-ex-02.png";
    Mat src = imread(filename, IMREAD_COLOR);

    // cv::resize(src, src, cv::Size(), 0.5, 0.5);

    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    //! [Load image]

    //! [Convert to grayscale]
    cvtColor( src, src, COLOR_BGR2GRAY );
    //! [Convert to grayscale]

    //! [Apply Histogram Equalization]
    Mat dst;
    equalizeHist( src, dst );
    //! [Apply Histogram Equalization]

    //! [Display results]
    imshow( "Source image", src );
    imshow( "Equalized Image", dst );
    //! [Display results]

    //! [Wait until user exits the program]
    waitKey();
    //! [Wait until user exits the program]

    return 0;

}