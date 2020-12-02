/* System Libraries */
#include <iostream>
#include <vector>
#include <algorithm>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv){
    /* Quaternion (q) Initialization */
    // Initialization by scalar values, (w, x, y, z)!
    Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    Quaterniond q2(-0.5, 0.4, -0.1, 0.2);

    print("Arbitrary Quaternions:");
    printQuaternion("q1: ", q1);
    printQuaternion("q2: ", q2);
    cout << endl;

    // Notes that quaternions need to be normalized before use! 
    q1.normalize();
    q2.normalize();

    print("Normalized Quaternions:");
    printQuaternion("q1: ", q1);
    printQuaternion("q2: ", q2);
    cout << endl;

    /* Translation Vectors (t) Initialization */
    Vector3d t1(0.3, 0.1, 0.1);   // Translation Vector from the Frame {w} origin pointing to the Frame {1} origin.
    Vector3d t2(-0.1, 0.5, 0.3);  // Translation Vector from the Frame {w} origin pointing to the Frame {2} origin.

    printVector("t1: ", t1);
    printVector("t2: ", t2);

    /* Transformation Matrix (T) Initialization */
    // T(q) initializes Rot. Matrix (R) with Quaternion q of the Transformation Matrix (T)
    Isometry3d T1w(q1);  // Transformation Matrix from Frame {w} to Frame {1}
    Isometry3d T2w(q2);  // Transformation Matrix from Frame {w} to Frame {2}
 
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);
    
    /* Coordinate Transformation */
    // Overload, Vector3d can multiply a Matrix4d
    Vector3d p1(0.5, 0, 0.2);  // Arbitrary point P in the Frame {1}
    
    // Get the coordinates of the point P in Frame {2}, based on the coordinates of P in Frame {1}
    Vector3d p2 = T2w*T1w.inverse()*p1;  // p2 = T2w*Tw1*p1 = T2w*inv(T1w)*p1

    printVector("p1: ", p1);
    printMatrix<Matrix4d>("T1w: ", T1w.matrix());
    printMatrix<Matrix4d>("T2w: ", T2w.matrix());
    printVector("p2: ", p2);

    return 0;
}