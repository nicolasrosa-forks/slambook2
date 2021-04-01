/* System Libraries */
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

/* ========================== */
/*  Eigen3/Sophus' Functions  */
/* ========================== */
template <typename TTypeEigenMat>
void printMatrix(const char text[], TTypeEigenMat mat){
    cout << text << endl;
    cout << mat << "\n" << "(" << mat.rows() << ", " << mat.cols() << ")" << endl << endl;
}

template <typename TTypeEigenVec>
void printVector(const char text[], TTypeEigenVec vec){
    cout << text << endl;
    cout << vec << "\n" << "(" << vec.size() << ",)" << endl << endl;
}

template <typename TTypeEigenQuat>
void printQuaternion(const char text[], TTypeEigenQuat quat){
    cout << text << quat.coeffs().transpose() << endl << endl;
}

/**
 * @brief Convert Normalized Coordinates to Pixel Coordinates (Image Plane, f=1)
 *
 * @param x Point2f in Normalized Coordinates, x=(x,y)=(X/Z, Y/Z)
 * @param K Intrinsic Parameters Matrix
 * @return Point2f in Pixel Coordinates Coordinates, p=(u,v)
 */
Vector2d cam2pixel(const Vector3d &P, const Matrix3d &K) {
    return Vector2d(
        K(0, 0)*P[0]/P[2] + K(0, 2),  // u = fx*x + cx
        K(1, 1)*P[1]/P[2] + K(1, 2)   // v = fy*y + cy
    );
}