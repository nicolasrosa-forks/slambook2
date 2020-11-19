/* System Libraries */
#include <iostream>
#include <ctime>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>    // Eigen Core
#include <eigen3/Eigen/Dense>   // Algebraic operations of dense matrices (inverse, eigenvalues, etc.)

using namespace std;
using namespace Eigen;

/* Global Variables */
#define MATRIX_SIZE 50

void printElapsed(const time_t begin_time){
    float elapsed_time = float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    
    std::cout << elapsed_time << " s" << endl;
}

/* =========================================================== */
/*  This program demonstrates the use of the basic Eigen type  */
/* =========================================================== */
int main(int argc, char **argv){
    /* ----- Eigen::Matrix Declaration ----- */
    // All vectors and matrices in Eigen are Eigen::Matrix, which is a template class.
    // Its first three parameters are: data type, row, column.
    Matrix<float, 2, 3> matrix_2x3;  // Declares a 2x3 float matrix, non-initialized (Garbage)
    
    cout << "matrix_2x3 (garbage):\n" << matrix_2x3 << endl << endl;

    /* ----- Eigen::Vector Declaration ----- */
    // At the same time, Eigen provides many built-in types via typedef, but the bottom layer is still Eigen::Matrix.
    // For example, Vector3d is essentially Eigen::Matrix<double, 3, 1>, which is a three-dimensional vector.
    Matrix<float, 3, 1> vd_3d;  // Declares a 3x1 float vector, non-initialized (Garbage)
    Vector3d v_3d;              // This is the same, non-initialized (Garbage)
    
    cout << "v_3d (garbage):\n" << v_3d << endl << endl;
    cout << "vd_3d (garbage):\n" << vd_3d << endl << endl;

    /* ----- Eigen::Matrix Initialization ----- */
    // Matrix3d is essentially Eigen::Matrix<double, 3, 3>.
    Matrix3d matrix_3x3 = Matrix3d::Zero();  // Declares a 3x3 double matrix, initilized to zero

    cout << "matrix_3x3 (zero):\n" << matrix_3x3 << endl << endl;
    
    // If you are not sure about the size of the matrix, you can use a matrix of dynamic size.
    // There are still many types of this, we doesn't list them one by one.
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;  // Declares a ?x? double matrix, empty
    MatrixXd matrix_x;                                // Simpler, empty
    
    cout << "matrix_dynamic:\n" << matrix_dynamic << endl << endl;
    cout << "matrix_x:\n" << matrix_x << endl << endl;

    // Here is the operation of the Eigen array
    // Input data (initialization)
    matrix_2x3 << 1, 2, 3, 4, 5, 6;

    cout << "matrix_2x3 (initilized):\n" << matrix_2x3 << endl << endl;
    
    /* ----- Eigen::Matrix Indexing ----- */
    // Use () to access elements in the matrix
    cout << "matrix_2x3 (initialized): " << endl;
    for (int i=0; i<2; i++){
        for (int j=0; j<3; j++)
            cout << matrix_2x3(i, j) << "\t";
            cout << endl;
    }
    cout << endl;

    // The matrix and vector are multiplied (actually still matrices and matrices)
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    /* ----- Eigen::Matrix Type Conversion ----- */
    // But in Eigen you can't mix two different types of matrices, the terms should be explicitly converted
    // Wrong: (double) result = (float) matrix_2x3 * (double) v_3d
    // Matrix<double, 2, 1> result1_wrong_types = matrix_2x3 * v_3d;  // Uncomment, it will throw an error!
    
    // Correct: (double) result = (double) matrix_2x3 * (double) v_3d
    Matrix<double, 2, 1> result1_correct_types = matrix_2x3.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1]=" << result1_correct_types.transpose() << endl << endl;

    // Correct: (float) result  = (float) matrix_2x3 * (float) vd_3d
    Matrix<float, 2, 1> result2_correct_types = matrix_2x3 * vd_3d; 
    cout << "[1,2,3;4,5,6]*[4,5,6]=" << result2_correct_types.transpose() << endl << endl;

    /* ----- Eigen::Matrix Dimensions ----- */
    // Also you can't misjudge the dimensions of the matrix
    // Try canceling the comments below to see what Eigen will report.
    // Matrix<double, 2, 3> result1_wrong_dimension = matrix_2x3.cast<double>() * v_3d; // Uncomment, it will throw an error!

    /* ----- Eigen::Matrix Operations ----- */
    matrix_3x3 = Matrix3d::Random();  // Random 3x3 Matrix

    cout << "Random matrix, M:\n" << matrix_3x3 << endl << endl;

    cout << "Addtion, M+10: \n" << matrix_3x3 + MatrixXd::Constant(3,3,10) << endl << endl;
    cout << "Subtractor, M-10: \n" << matrix_3x3 - MatrixXd::Constant(3,3,10) << endl << endl;
    cout << "Multiplication, Mx10: \n" << matrix_3x3*10.0 << endl << endl;
    cout << "Division, M/10: \n" << matrix_3x3/10.0 << endl << endl;

    cout << "Transpose, trans(M):\n" << matrix_3x3.transpose() << endl << endl;
    cout << "Summation, sum(M):\n" << matrix_3x3.sum() << endl << endl;
    cout << "Trace, tr(M):\n" << matrix_3x3.trace() << endl << endl;
    cout << "Inverse, inv(M):\n" << matrix_3x3.inverse() << endl << endl;
    cout << "Determinant, det(M):\n" << matrix_3x3.determinant() << endl << endl;
    
    /* ----- Eigenvalues ----- */
    // Real symmetric matrix can guarantee successful diagonalization
    const clock_t begin_time = clock();
    cout << "Calculating Eigenvalues..." << endl;
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_3x3.transpose() * matrix_3x3);
    printElapsed(begin_time);

    cout << endl;
    cout << "Eigen values:\n" << eigen_solver.eigenvalues() << endl << endl;
    cout << "Eigen vectors:\n" << eigen_solver.eigenvectors() << endl << endl;

    return 0;
}