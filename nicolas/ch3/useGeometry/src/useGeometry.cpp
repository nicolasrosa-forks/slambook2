/* System Libraries */
#include <iostream>
#include <cmath>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;


/* ================================================================ */
/*  This program demonstrates how to use the Eigen geometry module  */
/* ================================================================ */
// The Eigen/Geometry module provides a variety of rotation and translation representations

// Hint:
// Rotation matrix (3×3): Eigen::Matrix3d.
// Rotation vector (3×1): Eigen::AngleAxisd.
// Euler angle (3×1): Eigen::Vector3d.
// Quaternion (4×1): Eigen::Quaterniond.
// Euclidean transformation matrix (4×4): Eigen::Isometry3d.
// Affine transform (4×4): Eigen::Affine3d.
// Perspective transformation (4×4): Eigen::Projective3d.

int main(int argc, char** argv){
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

    cout << "v1=[1,0,0] after rotation (by angle axis): " << v_rot1.transpose() << endl;
    cout << "v2=[0,1,0] after rotation (by angle axis): " << v_rot2.transpose() << endl;
    cout << "v3=[0,0,1] after rotation (by angle axis): " << v_rot3.transpose() << endl << endl;

    // Rotation by Matrix
    v_rot1 = R*v1;
    v_rot2 = R*v2;
    v_rot3 = R*v3;

    cout << "v1=[1,0,0] after rotation (by matrix): " << v_rot1.transpose() << endl;
    cout << "v2=[0,1,0] after rotation (by matrix): " << v_rot2.transpose() << endl;
    cout << "v2=[0,0,1] after rotation (by matrix): " << v_rot3.transpose() << endl << endl;

    /* 4. Euler angles */
    // You can convert the rotation matrix directly into Euler angles
    Vector3d euler_angles = R.eulerAngles(2, 1, 0);  // RPY Angles, ZYX order

    cout << "rpy=[yaw, pitch, roll]: " << euler_angles.transpose() << endl << endl;

    /* 5. Euclidean transformation matrix using Eigen::Isometry */
    Isometry3d T = Isometry3d::Identity();  //Although called 3d, it's essentially a Matrix4d (4x4)
    T.rotate(n);                            // Rotate according to the rotation vector n.
    T.pretranslate(Vector3d(1,3,4));        // Set the translation vector, t=(1,3,4)

    printMatrix<Matrix4d>("T:", T.matrix());

    // Use the transformation matrix for coodinate transformation
    Vector3d v_transformed = T*v1;  // Equivalent to R*v+t

    cout << "v1=[1,0,0] after transformation T(R, t): " << v_transformed.transpose() << endl << endl;

    /* 6. For affine and projective transformations, use Eigen::Affine3d and Eigen::Projective3d.
    //TODO

    /* Quaternions */
    // You can assign AngleAxis directly to quaternions, and vice versa
    Quaterniond q1 = Quaterniond(n);  // Quaternion(q1) initialized from Rotation Vector(n)

    // Note that the order of coeffs is (x, y, z, w), w is the real part, the first three are the imaginary part
    Vector4d q1_coeffs = q1.coeffs();
    double q1_real_part = q1.w();
    Vector3d q1_imag_part = q1.vec();

    printQuaternion("q1[s1, v1]: ", q1);
    cout << "s1: " << q1_real_part << endl;  // Or, w
    cout << "v1: " << q1_imag_part.transpose() << endl;  // Or, [x, y, z]

    printMatrix<Matrix3d>("R from q1:", q1.matrix());  // Converts Quaternion created by AngleAxis to Rotation Matrix

    // Can also assign a rotation matrix to it
    Quaterniond q2 = Quaterniond(R);  // Quaternion(q2) initialized from Rotation Matrix(R)

    Vector4d q2_coeffs = q2.coeffs();
    double q2_real_part = q2.w();
    Vector3d q2_imag_part = q2.vec();

    printQuaternion("q2[s2, v2]: ", q2);
    cout << "s2: " << q2_real_part << endl;  // Or, w
    cout << "v2: " << q2_imag_part.transpose() << endl << endl;  // Or, [x, y, z]

    /* =================================== */
    /*  Rotate a vector with a quaternion  */
    /* =================================== */
    // [Method 1] By operator overloading in C++, quaternions and 3D Vectors can directly be multiplied.
    // [Method 2] However, mathematically, the vector needs to be converted into an imaginary quaternion like shown in the book, and then quaternion multiplication is used for calculation.

    /* ----- Method 1, returns a Vector3d ----- */
    // Using overloaded multiplication
    Vector3d v1_new = q1*v1;  // Note that the math is: v' = q.v, (Rot. Quaternion*Vector)

    cout << "[Method 1]" << endl;
    printQuaternion("q1: ", q1);
    cout << "v1: " << v1.transpose() << endl;
    printQuaternion("inv(q1): ", q1.inverse());
    cout << "v1_new: " << v1_new.transpose() << endl << endl;

    /* ----- Method 2, returns a Quaterniond ----- */
    // Option 1: Quaternion Initialization by Vector4d, (x, y, z, w)!
    // Vector4d v1_aux{1, 0, 0, 0};  // [v1, 0]
    // Quaterniond q_v1 = Quaterniond(v1_aux);

    // Option 2: Quaternion Initialization by scalar values, (w, x, y, z)!
    Quaterniond q_v1 = Quaterniond(0, 1, 0, 0);

    // Select different initialization options to see that the declaration way matters. It changes the Quaternion coefficients!!!
    cout << "[Method 2]" << endl;
    cout << "x: " << q_v1.x() << endl;
    cout << "y: " << q_v1.y() << endl;
    cout << "z: " << q_v1.z() << endl;
    cout << "w: " << q_v1.w() << endl << endl;

    Quaterniond q_v1_new = q1*q_v1*q1.inverse();  // Note that the math is: v' = q.v.inv(q), (Rot. Quaternion * Quat. of Vector*Inv. of Rot. Quaternion)

    printQuaternion("q1: ", q1);
    printQuaternion("q_v1: ", q_v1);
    printQuaternion("inv(q1): ", q1.inverse());
    printQuaternion("q_v1_new: ", q_v1_new);
    cout << "v1_new: " <<  q_v1_new.vec().transpose() << endl << endl;

    return 0;
}