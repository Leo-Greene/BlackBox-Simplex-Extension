/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fitness_bb.h
 *
 * Code generation for function 'fitness_bb'
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
real_T fitness_bb(const emlrtStack *sp, const real_T pos[14], const real_T vel
                  [14], real_T params_n, real_T params_ws_bb, real_T
                  params_w_orient);

/* End of code generation (fitness_bb.h) */
