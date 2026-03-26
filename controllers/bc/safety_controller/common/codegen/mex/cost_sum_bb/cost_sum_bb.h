/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * cost_sum_bb.h
 *
 * Code generation for function 'cost_sum_bb'
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
#include "cost_sum_bb_types.h"

/* Function Declarations */
real_T cost_sum_bb(const emlrtStack *sp, const real_T u[140], real_T pos[14],
                   real_T vel[14], const struct0_T *params);

/* End of code generation (cost_sum_bb.h) */
