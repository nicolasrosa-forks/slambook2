#ifndef LIBUTILS_H_
#define LIBUTILS_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

/* ================== */
/*  Functions Scopes  */
/* ================== */
void print(char var[]);

template <typename TTypeMat>
void printMatrix(const char text[], TTypeMat mat);

template <typename TTypeVec>
void printVector(const char text[], TTypeVec vec);

template <typename TTypeQuat>
void printQuaternion(const char text[], TTypeQuat quat);

#include "libUtils.cpp"

#endif