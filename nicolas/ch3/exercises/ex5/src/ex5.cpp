#include <iostream>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>

/* Custom Libraries */
#include "../../../include/libUtils.h"

using namespace std;
using namespace Eigen;


int main(int argc,char** argv){
    Matrix<double, 50, 50> M = MatrixXd::Random(50,50);
    Matrix3d I = Matrix3d::Identity();

    printMatrix<MatrixXd>("M: ", M);
    printMatrix<Matrix3d>("I: ", I);

    printMatrix<MatrixXd>("M(0:2, 0:2): ", M.block<3,3>(0,0));
    
    I = M.block<3,3>(0,0);

    printMatrix<Matrix3d>("I': ", I);

    return 0;
}