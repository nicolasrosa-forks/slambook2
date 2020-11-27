/* System Libraries */
#include <iostream>
#include <cmath>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace Eigen;

/* Global Variables */
void print(char var[]){
    cout << var << endl;
}

template <typename TTypeMat>
void printMatrix(char text[], TTypeMat mat){
    cout << text << endl;
    cout << mat << "\n" << "(" << mat.rows() << ", " << mat.cols() << ")" << endl << endl;
}

template <typename TTypeVec>
void printVector(char text[], TTypeVec vec){
    cout << text << endl;
    cout << vec << "\n" << "(" << vec.size() << ",)" << endl << endl;
}

/* ================================================================ */
/*  This program demonstrates how to use the Eigen geometry module  */
/* ================================================================ */
// The Eigen/Geometry module provides a variety of rotation and translation representations
int main(int argc, char **argv){
    /* 1. Rotation Matrix */
    // 3D rotation matrix can be declared directly using Matrix3d (3x3, double) or Matrix3f (3x3, float)
    Matrix3d R = Matrix3d::Identity();  // Rotation Matrix
    printMatrix<Matrix3d>("R:", R);

    /* 2. Angle-Axis */
    // The rotation vector uses AngleAxis, the underlying layer is not directly Matrix, 
    // but the operation can be treated as a matrix (because the operator is overloaded)
    AngleAxisd n(M_PI_4, Vector3d(0, 0, 1));  // Rotation Vector (n), rotate pi/4 rad (45 deg)  along the Z-axis

    // Rot. Vector -> Rot. Matrix (Z-axis)
    cout.precision(3);
    printMatrix<Matrix3d>("n:", n.matrix()); 

    // can also be assigned directly    
    R = n.toRotationMatrix();  // or just n.matrix()
    printMatrix<Matrix3d>("R:", R);

    /* 3. Coordinate transformation */
    Vector3d v1(1, 0, 0);  // Arbitrary Vector in X-axis direction
    Vector3d v2(0, 1, 1);  // Arbitrary Vector in Y-axis direction
    Vector3d v3(0, 0, 1);  // Arbitrary Vector in Z-axis direction (Coordinates won't change, vector at same direction of the rotation)

    // Rotation by Angle-Axis
    Vector3d v_rot1 = n*v1;
    Vector3d v_rot2 = n*v2;
    Vector3d v_rot3 = n*v3;
    
    cout << "v1=(1,0,0) after rotation (by angle axis): " << v_rot1.transpose() << endl;
    cout << "v2=(0,1,0) after rotation (by angle axis): " << v_rot2.transpose() << endl;
    cout << "v3=(0,0,1) after rotation (by angle axis): " << v_rot3.transpose() << endl << endl;

    // Rotation by Matrix
    v_rot1 = R*v1;
    v_rot2 = R*v2;
    v_rot3 = R*v3;
    
    cout << "v1=(1,0,0) after rotation (by matrix): " << v_rot1.transpose() << endl;
    cout << "v2=(0,1,0) after rotation (by matrix): " << v_rot2.transpose() << endl;
    cout << "v2=(0,0,1) after rotation (by matrix): " << v_rot3.transpose() << endl << endl;

    /* 4. Euler angles */
    // You can convert

    return 0;
}