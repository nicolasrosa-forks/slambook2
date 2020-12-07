#include <iostream>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>   // Algebraic operations of dense matrices (inverse, eigenvalues, etc.)

/* Custom Libraries */
#include "../../../include/libUtils.h"

using namespace std;
using namespace Eigen;


int main(int argc,char** argv){
    Matrix3d A = Matrix3d::Random(3, 3);
    Vector3d b = Vector3d::Random(3);

    A = A*A.transpose();  // Make Sure, the generated Matrix is Positive-Definite

    printMatrix<Matrix3d>("A: ", A);
    printVector<Vector3d>("b: ", b);

    /* Solving the system Ax = b */
    // Option 1: x = inv(A)*b
    Matrix<double, 3, 1> x = A.inverse()*b;

    print("[Option 1]: inv(A)*b");
    printMatrix<MatrixXd>("inv(A): ", A.inverse());
    printMatrix<MatrixXd>("x: ", x);

    // Option 2: QR Decomposition
    x = A.colPivHouseholderQr().solve(b);

    print("[Option 2]: QR Decomposition");
    printMatrix<MatrixXd>("x: ", x);
    
    // Option 3: Cholesky Decomposition
    x = A.ldlt().solve(b);

    print("[Option 3] Cholesky Decomposition");
    printMatrix<MatrixXd>("x: ", x);

    return 0;
}