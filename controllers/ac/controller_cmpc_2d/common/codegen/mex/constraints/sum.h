/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * sum.h
 *
 * Code generation for function 'sum'
 *
 */

#pragma once

/* Include files */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include "emlrt.h"
#include "rtwtypes.h"
#include "constraints_types.h"

/* Function Declarations */
void sum(const emlrtStack *sp, const emxArray_real_T *x, real_T y_data[],
         int32_T y_size[3]);

/* End of code generation (sum.h) */
