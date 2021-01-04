/* System Libraries */
#include <iostream>
#include <cmath>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Sophys Libraries */
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"


/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

/* =============================================================================== */
/*  This program demonstrates the SO(3) and SE(3) operations in the Sophus library */
/* =============================================================================== */
int main(int argc, char** argv){
    // Rotate 90 degrees along Z-axis
    Matrix3d R = AngleAxisd(M_PI_2, Vector3d(0, 0, 1)).toRotationMatrix();  // By Rotation Matrix (R);
    Quaterniond q(R);  // Or by Rotation quaternion (q).

    printMatrix<Matrix3d>("R: ", R);
    printQuaternion<Quaterniond>("q: ", q);

    /* ----------------- */
    /*  SO(3) Lie Group  */
    /* ----------------- */
    // Declaration
    Sophus::SO3d SO3_R(R);  // Sophus::SO3d can be constructed from the rotation matrix R;
    Sophus::SO3d SO3_q(q);  // or the quaternion q.

    // They are equivalent of course
    printMatrix<Matrix3d>("SO(3) from R: ", SO3_R.matrix());
    printMatrix<Matrix3d>("SO(3) from q: ", SO3_q.matrix());
    print("They are equal!");
    cout << endl;

    // Use the logarithmic map to get the 𝖘𝖔(3) Lie algebra, which is a 3D rotation vector (𝜙).
    // Reminder: 𝜙 = vee(ln(R)), SO(3) -> 𝖘𝖔(3)
    Vector3d so3_phi = SO3_R.log();
    Matrix3d so3_phi_hat = Sophus::SO3d::hat(so3_phi);

    printVector<Vector3d>("so3_phi: ", so3_phi);

    // We also can use hat and vee operators:
    printMatrix<Matrix3d>("so3_phi^: ", so3_phi_hat);                          // Hat is from Rotation Vector (𝜙) to Skew-Symmetric Matrix (𝜙^) in Lie Algebra, 𝖘𝖔(3).
    printVector<Vector3d>("vee(so3_phi^): ", Sophus::SO3d::vee(so3_phi_hat));  // Inversely from Skew-Symmetric Matrix (𝜙^) to Rotation Vector (𝜙) in Lie Algebra, 𝖘𝖔(3).

    // To save back again to SO(3) , we have to use the exponential:
    printMatrix<Matrix3d>("R, SO3_R: ", Sophus::SO3d::exp(so3_phi).matrix());  // When programming, we can just use 𝜙, but according to book it should be 𝜙^.

    // Update by using the perturbation model.
    Vector3d small_so3_phi(1e-4, 0, 0);  // This is a small update (rotation vector in Lie Algebra)
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(small_so3_phi)*SO3_R;  // Update: Lie Group (SO(3)) Left Multiplication, ΔR.R = exp(δ𝜙).R
    
    printVector<Vector3d>("small_so3_phi: ", small_so3_phi);
    printMatrix<Matrix3d>("SO(3) updated: ", SO3_updated.matrix());

    /* ----------------- */
    /*  SE(3) Lie Group  */
    /* ----------------- */
    // Translation Vector Declaration
    Vector3d t(1, 0, 0);  // Translates 1 along X-axis
    
    // SE(3) Declaration, also known as Transformation matrix (T) or Pose
    // It can be initialized ...
    Sophus::SE3d SE3_Rt(R, t);  // ... from R, t
    Sophus::SE3d SE3_qt(q, t);  // or  from q, t

    printMatrix<Matrix4d>("SE(3) from R,t: ", SE3_Rt.matrix());
    printMatrix<Matrix4d>("SE(3) from q,t: ", SE3_qt.matrix());
    
    // Use the logarithmic map to get the 𝖘𝖊(3) Lie Algebra, which is a 6D vector.
    // Reminder: 𝛏 = vee(ln(T)), SE(3) -> 𝖘𝖊(3)
    typedef Eigen::Matrix<double, 6, 1> Vector6d;  // Since Eigen doesn't have Vector6d, we need to create it.

    Vector6d se3_xi = SE3_Rt.log();                   // 𝛏  = [ρ, 𝜙]'
    Matrix4d se3_xi_hat = Sophus::SE3d::hat(se3_xi);  // 𝛏^ = [𝜙^ ρ; 0' 0]

    printVector<Vector6d>("se3_xi: ", se3_xi);  // The output shows that Sophus puts the translation (ρ) at first in 𝖘𝖊(3), then rotation (𝜙).

    // We also can use hat and vee operators:
    printMatrix<Matrix4d>("se3_xi^: ", se3_xi_hat);
    printVector<Vector6d>("vee(se3_xi^): ", Sophus::SE3d::vee(se3_xi_hat));

    // To save back again to SE(3), we have to use the exponential:
    printMatrix<Matrix4d>("T, SE3_Rt: ", Sophus::SE3d::exp(se3_xi).matrix());  // When programming, we can just use 𝛏, but according to book it should be 𝛏^.
    
    // Finally, update by using the perturbation model.
    Vector6d small_se3_xi;
    small_se3_xi << 1e-4, 0, 0, 0, 0, 0;

    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(small_se3_xi)*SE3_Rt;  // Update: Lie Group (SE(3)) Left Multiplication, ΔT.T = exp(δ𝛏).T

    printVector<Vector6d>("small_se3_xi: ", small_se3_xi);
    printMatrix<Matrix4d>("SE(3) updated: ", SE3_updated.matrix());

    
    return 0;
}