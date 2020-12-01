/* System Libraries */
#include <iostream>

using namespace std;

/* =========== */
/*  Functions  */
/* =========== */
void print(char var[]){
    cout << var << endl;
}

template <typename TTypeMat>
void printMatrix(const char text[], TTypeMat mat){
    cout << text << endl;
    cout << mat << "\n" << "(" << mat.rows() << ", " << mat.cols() << ")" << endl << endl;
}

template <typename TTypeVec>
void printVector(const char text[], TTypeVec vec){
    cout << text << endl;
    cout << vec << "\n" << "(" << vec.size() << ",)" << endl << endl;
}

template <typename TTypeQuat>
void printQuaternion(const char text[], TTypeQuat quat){
    cout << text << quat.coeffs().transpose() << endl;
}