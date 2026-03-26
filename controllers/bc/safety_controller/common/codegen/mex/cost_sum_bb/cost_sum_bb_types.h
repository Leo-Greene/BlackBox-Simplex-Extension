/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * cost_sum_bb_types.h
 *
 * Code generation for function 'cost_sum_bb_types'
 *
 */

#pragma once

/* Include files */
#include "rtwtypes.h"

/* Type Definitions */
#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T

struct emxArray_real_T
{
  real_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_real_T*/

#ifndef typedef_emxArray_real_T
#define typedef_emxArray_real_T

typedef struct emxArray_real_T emxArray_real_T;

#endif                                 /*typedef_emxArray_real_T*/

#ifndef typedef_struct0_T
#define typedef_struct0_T

typedef struct {
  real_T n;
  real_T dt;
  real_T ct;
  real_T h_ac;
  real_T h_bc;
  real_T steps;
  real_T amax;
  real_T vmax;
  real_T dmin;
  real_T diameter;
  real_T switch_step;
  real_T ws;
  real_T wt;
  real_T ws_bb;
  real_T w_orient;
  real_T predator;
  real_T pFactor;
  real_T pred_radius;
  real_T wp;
  real_T tgt[14];
} struct0_T;

#endif                                 /*typedef_struct0_T*/

/* End of code generation (cost_sum_bb_types.h) */
