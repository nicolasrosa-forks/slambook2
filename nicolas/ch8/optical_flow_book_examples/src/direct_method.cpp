/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>

/* OpenCV Libraries */
#include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/calib3d/calib3d.hpp>

/* Sophus Libraries */
#include <sophus/se3.hpp>

/* Boost Libraries */
#include <boost/format.hpp>

/* Pangolin Libraries */
#include <pangolin/pangolin.h>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"
#include "../../../common/libUtils_opencv.h"

using namespace std;
using namespace cv;

/* Global Variables */
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

string left_file = "../../images/left.png"; // FIXME: I think this image is the 000000.png
string disparity_file = "../../images/disparity.png";
boost::format fmt_others("../../images/%06d.png"); // Other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// Camera Intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double b = 0.573; // Baseline (Stereo Camera)

/* =================== */
/*  Class Declaration  */
/* =================== */

/**
 * @brief Class for accumulate Jacobians in parallel
 * 
 */
class JacobianAccumulator {
public:
    // Constructor
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_, //FIXME: Use &?
        Sophus::SE3d &T21_) : img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_)
    {
        p2_proj = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /* Accumulate jacobians in a range */
    void accumulate_jacobian(const cv::Range &range);

    /* Get Hessian Matrix */
    Matrix6d getHessian() const { return H; }

    /* Get Bias Vector */
    Vector6d getBias() const { return b; }

    /* Get Total Cost */
    double getTotalCost() const { return cost; }

    /* Get Projected Points */
    VecVector2d getProjectedPoints() const { return p2_proj; }

    /* Reset H, b, and cost */
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0.0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref; //FIXME: Use &?
    Sophus::SE3d &T21;

    VecVector2d p2_proj;            // Projected Points
    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0.0;
};

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
/**
 * @brief Pose estimation using Direct Method (Multi-layer)
 * 
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref, //FIXME: Use &?
    Sophus::SE3d &T21);

/**
 * @brief Pose estimation using Direct Method (Single-layer)
 * 
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref, //FIXME: Use &?
    Sophus::SE3d &T21);

/** Bilinear Interpolation
 * @brief Get a grayscale value from reference image (bilinear interpolation)
 * 
 * @param img input image
 * @param x x-coordinate of the center pixel
 * @param y y-coordinate of the center pixel
 * @return the interpolated value of this pixel
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    /* Boundary check */
    if (x < 0)
        x = 0; // Avoid negative x-axis coordinates
    if (y < 0)
        y = 0; // Avoid negative y-axis coordinates
    if (x >= img.cols)
        x = img.cols - 1; // Avoid positive x-axis coordinates outside image width
    if (y >= img.rows)
        y = img.rows - 1; // Avoid positive y-axis coordinates outside image height

    uchar *data = &img.data[int(y) * img.step + int(x)];

    float xx = x - floor(x);
    float yy = y - floor(y);

    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to implement a sparse direct method of binocular cameras (i.e, vSLAM with Known Depth!). */
int main(int argc, char **argv) {
    cout << "[direct_method] Hello!" << endl << endl;

    /* Load the images */
    cv::Mat left_img = cv::imread(left_file, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat disparity_img = cv::imread(disparity_file, CV_LOAD_IMAGE_GRAYSCALE);

    /* Initialization */
    // Let's randomly pick pixels in the first image and generate some 3D points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // Generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        // Don't pick pixels close to the boarder
        int x = rng.uniform(boarder, left_img.cols - boarder);
        int y = rng.uniform(boarder, left_img.rows - boarder);

        // Use the camera intrinsics parameters to get the depth from the disparity image.
        // Note: Remember that in OpenCV, the coordinates axes for retriving the pixel value are inverted.
        int disparity = disparity_img.at<uchar>(y, x); // Disp(x, y)
        double depth = (fx * b) / disparity;           // D(x, y)

        depth_ref.push_back(depth);           // {D(x, y)_i}
        pixels_ref.push_back(Vector2d(x, y)); // {[x, y]_i}
    }

    // Estimates 01~05.png's pose using this information.
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) { // [1, 5]
        cv::Mat img = cv::imread((fmt_others % i).str(), CV_LOAD_IMAGE_GRAYSCALE);

        // Try single layer by uncommenting the following line
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        // DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);  //TODO
    }

    /* --------- */
    /*  Results  */
    /* --------  */
    /* Display Images */
    // imshow("image1", image1);
    // imshow("image2", image2);
    imshow("left", left_img);
    imshow("disparity", disparity_img);
    cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    cout << "Done." << endl;

    return 0;
}

/* ========================= */
/*  Function Implementation  */
/* ========================= */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref, //FIXME: Use &?
    Sophus::SE3d &T21){

    /* Initialization */
    const int iterations = 10;
    double cost = 0.0, lastCost = 0.0;

    Timer t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()), std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.getHessian();
        Vector6d b = jaco_accu.getBias();

        /* ----- Solve! ----- */
        // Solve the Linear System A*x=b, H(x)*∆x = g(x)
        Vector6d update = H.ldlt().solve(b); // ∆x = δξ (Lie Algebra)
        cost = jaco_accu.getTotalCost();

        // Check Solution
        if (std::isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }

        /* Stopping Criteria */
        // If the cost increased, the update was not good, then break.
        if (iter > 0 && cost > lastCost)
        { // FIXME: I think the correct one is "cost >= lastCost"
            cout << "\ncost: " << cost << " >= lastCost: " << lastCost << ", break!" << endl;
            break;
        }

        /* ----- Update ----- */ // FIXME: Esta parte estava antes dos if acima, mas acredito que o correto seja fazer o update após os ifs
        // Left multiply T by a disturbance quantity exp(δξ)
        T21 = Sophus::SE3d::exp(update) * T21; // Left perturbation

        if (update.norm() < 1e-3)
        { // Method converged!
            break;
        }

        lastCost = cost;
        cout << "it: " << iter << ",\tcost: " << std::setprecision(12) << cost << ",\tupdate: " << update.transpose() << endl;
    }

    printMatrix<Matrix4d>("T21: ", T21.matrix());
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Direct method (Single-layer): ", t1, t2);

    /* Plot the projected pixels */
    Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.getProjectedPoints();

    for (size_t i = 0; i < px_ref.size(); ++i)
    {                               // FIXME: pq ++i, e não i++
        auto p_ref = px_ref[i];     // p1_i
        auto p_cur = projection[i]; // p2^_i

        if (p_cur[0] > 0 && p_cur[1] > 0)
        { // x, y
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]), cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("Current", img2_show);
    cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) { // cv::Range(0, px_ref.size())
    /* Parameters */
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {
        /* Compute the projection in the second image */
        Eigen::Vector3d x1_ref = Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);  // p1_i->x1_i, x1 = [X/Z, Y/Z, 1]
        Eigen::Vector3d point_ref = depth_ref[i] * x1_ref;                                                // P1_i = [X, Y, Z]
        Eigen::Vector3d point_cur = T21 * point_ref;                                                      // P2^_i = T21*.P1_i

        if (point_cur[2] < 0) // invalid depth
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx; // u = fx*(X/Z)+cx
        float v = fy * point_cur[1] / point_cur[2] + cy; // v = fy*(Y/Z)+cy

        if (u < half_patch_size || u > img2.cols - half_patch_size ||
            v < half_patch_size || v > img2.rows - half_patch_size)
            continue;

        p2_proj[i] = Eigen::Vector2d(u, v); // p2^ = [u, v]
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
               Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;

        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {
                double error =  GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                                GetPixelValue(img2, u + x, v + y);

                Eigen::Vector2d J_img_pixel; // ∂I2/∂u
                Matrix26d J_pixel_xi;        // ∂u/∂δξ

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                // Total Jacobian, J = −(∂I2/∂u)*(∂u/∂δξ)
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose(); // 6x1

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good)
    {
        // Set Hessian, bias and cost_tmp
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}