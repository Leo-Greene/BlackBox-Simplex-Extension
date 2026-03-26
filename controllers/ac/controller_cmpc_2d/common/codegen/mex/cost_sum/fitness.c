/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fitness.c
 *
 * Code generation for function 'fitness'
 *
 */

/* Include files */
#include "fitness.h"
#include "cost_sum.h"
#include "cost_sum_data.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo i_emlrtRSI = { 6,   /* lineNo */
  "fitness",                           /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/controller_cmpc_2d/fitness.m"/* pathName */
};

static emlrtBCInfo b_emlrtBCI = { 1,   /* iFirst */
  7,                                   /* iLast */
  8,                                   /* lineNo */
  49,                                  /* colNo */
  "pos",                               /* aName */
  "separation_polynomial",             /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/controller_cmpc_2d/cost_functions/separation_polynomial.m",/* pName */
  0                                    /* checkKind */
};

/* Function Definitions */
real_T fitness(const emlrtStack *sp, const real_T pos[14], real_T params_ws,
               real_T params_wt, const real_T params_tgt[14])
{
  real_T b_sp;
  int32_T i;
  int32_T b_i;
  int32_T j;
  int32_T xpageoffset;
  int32_T i1;
  real_T circle_tgt;
  real_T z1[14];
  real_T z1_idx_0;
  real_T y[7];
  emlrtStack st;
  st.prev = sp;
  st.tls = sp->tls;

  /*  cmpc - fitness  */
  /*  separation */
  st.site = &i_emlrtRSI;

  /*  cmpc Given an pos -> 2 x n matrix, res is sum of inverse of squared pairwise distances.  */
  /*  */
  b_sp = 0.0;
  for (i = 0; i < 6; i++) {
    b_i = 5 - i;
    for (j = 0; j <= b_i; j++) {
      xpageoffset = (i + j) + 2;
      if (xpageoffset > 7) {
        emlrtDynamicBoundsCheckR2012b(xpageoffset, 1, 7, &b_emlrtBCI, &st);
      }

      i1 = i << 1;
      xpageoffset = (xpageoffset - 1) << 1;
      circle_tgt = pos[i1] - pos[xpageoffset];
      z1_idx_0 = circle_tgt * circle_tgt;
      circle_tgt = pos[i1 + 1] - pos[xpageoffset + 1];
      b_sp += 1.0 / (z1_idx_0 + circle_tgt * circle_tgt);
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b(&st);
      }
    }

    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(&st);
    }
  }

  b_sp /= 21.0;

  /*  cohesion */
  /*  asd = sum_sq_distances( pos ); */
  /*  target cost */
  /* SUM_SQ_DISTANCES_TGT Summary of this function goes here */
  for (xpageoffset = 0; xpageoffset < 14; xpageoffset++) {
    circle_tgt = pos[xpageoffset] - params_tgt[xpageoffset];
    z1[xpageoffset] = circle_tgt * circle_tgt;
  }

  for (i = 0; i < 7; i++) {
    xpageoffset = i << 1;
    y[i] = z1[xpageoffset] + z1[xpageoffset + 1];
  }

  circle_tgt = y[0];
  for (xpageoffset = 0; xpageoffset < 6; xpageoffset++) {
    circle_tgt += y[xpageoffset + 1];
  }

  /*  res = asd + 1000000 * spC; */
  return params_wt * circle_tgt + params_ws * b_sp;
}

/* End of code generation (fitness.c) */
