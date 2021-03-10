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

double getElapsedTime_ms(const time_t t_start){
    double elapsed_time = 1000*(clock () - t_start) / (double) CLOCKS_PER_SEC;

    return elapsed_time;
}

/* =========================================================== */
/*  This program demonstrates the use of the basic Eigen type  */
/* =========================================================== */
int main(int argc, char** argv){
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

    cout << "matrix_dynamic (empty):\n" << matrix_dynamic << endl << endl;
    cout << "matrix_x (empty):\n" << matrix_x << endl << endl;

    // Here is the operation of the Eigen array
    // Input data (initialization)
    matrix_2x3 << 1, 2, 3, 4, 5, 6;

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
    // Matrix<double, 2, 3> result1_wrong_dimension = matrix_2x3.cast<double>() * v_3d;  // Uncomment, it will throw an error!

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
    clock_t t_start = clock();
    cout << "Calculating Eigenvalues..." << endl;
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_3x3.transpose() * matrix_3x3);
    double t_elapsed = getElapsedTime_ms(t_start);
    cout << "Time for Solving Eigenvalues: " << t_elapsed << "ms" << endl << endl;

    cout << "Eigen values:\n" << eigen_solver.eigenvalues() << endl << endl;
    cout << "Eigen vectors:\n" << eigen_solver.eigenvectors() << endl << endl;

    /* ----- Solving Equations ----- */
    // We solve the equation of matrix_NN * x = v_Nd
    // The size of N is defined in the previous macro, which is generated by a
    // random number Direct inversion s the most direct, but the amount of
    // inverse operations is large.

    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NxN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    // For any real invertible matrix A, the product M.T*M is a positive definite matrix.
    // So let's guarantee that our matrix is positive definite,  x.T*M*x >= 0 for all x in R^n.
    matrix_NxN = matrix_NxN*matrix_NxN.transpose();

    cout << "matrix_NxN (random):\n" << matrix_NxN << endl << endl;
    cout << "v_Nd (random):\n" << v_Nd.transpose() << endl << endl;

    // 1. Direct inversion
    // x = inv(M)*v
    t_start = clock();  // reset timer
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NxN.inverse() * v_Nd;
    t_elapsed = getElapsedTime_ms(t_start);

    cout << "[Method 1] Time of Normal Inverse: " << t_elapsed << "ms" << endl;
    cout << "x:\n" << x.transpose() << endl << endl;

    // 2. QR Decomposition
    // Usually solved by matrix decomposition, such as QR decomposition, the speed will be much faster
    t_start = clock();  //reset timer
    x = matrix_NxN.colPivHouseholderQr().solve(v_Nd);
    t_elapsed = getElapsedTime_ms(t_start);

    cout << "[Method 2] Time of QR decomposition: " << t_elapsed << "ms" << endl;
    cout << "x:\n" << x.transpose() << endl << endl;

    // 3. Cholesky Decomposition
    // For positive definite matrices, you can also use Cholesky decomposition to solve equations.
    t_start = clock();  // reset timer
    x = matrix_NxN.ldlt().solve(v_Nd);
    t_elapsed = getElapsedTime_ms(t_start);

    cout << "[Method 3] Time of Cholesky decomposition: " << t_elapsed << "ms" << endl;
    cout << "x:\n" << x.transpose() << endl << endl;

    return 0;
}