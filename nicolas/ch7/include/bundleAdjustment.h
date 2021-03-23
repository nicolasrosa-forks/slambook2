#ifndef bundleAdjustment_H_
#define bundleAdjustment_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

#include "bundleAdjustment.cpp"

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
void bundleAdjustmentGaussNewton();

void bundleAdjustmentG2O();

#endif