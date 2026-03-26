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
#include "mwmathutil.h"
#include "predator_avoidance.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo j_emlrtRSI = { 14,  /* lineNo */
  "fitness",                           /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/fitness.m"/* pathName */
};

static emlrtRSInfo k_emlrtRSI = { 27,  /* lineNo */
  "fitness",                           /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/fitness.m"/* pathName */
};

static emlrtRSInfo l_emlrtRSI = { 28,  /* lineNo */
  "fitness",                           /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/fitness.m"/* pathName */
};

static emlrtBCInfo c_emlrtBCI = { 1,   /* iFirst */
  10,                                  /* iLast */
  8,                                   /* lineNo */
  49,                                  /* colNo */
  "pos",                               /* aName */
  "separation_polynomial",             /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/separation_polynomial.m",/* pName */
  0                                    /* checkKind */
};

/* Function Definitions */
real_T fitness(const emlrtStack *sp, const real_T pos[20], real_T params_n,
               real_T params_ws, real_T params_wc, real_T params_wt, real_T
               params_predator, real_T params_wp, const real_T params_tgt[20])
{
  real_T res;
  real_T b_sp;
  int32_T i;
  int32_T b_i;
  int32_T j;
  int32_T xpageoffset;
  int32_T i1;
  real_T circle_tgt;
  real_T z1_idx_0;
  real_T z1[20];
  real_T y[10];
  emlrtStack st;
  st.prev = sp;
  st.tls = sp->tls;

  /*  cmpc - fitness  */
  /*  Obstacle Avoidance */
  /*  rects = params.rects; */
  /*  obst = obstacle_cost(pos, rects, params); */
  /*  Target seeking */
  /*  tar = [60; 50]; */
  /*  tgt = target(pos, tar, params); */
  /*  separation */
  st.site = &j_emlrtRSI;

  /*  cmpc Given an pos -> 2 x n matrix, res is sum of inverse of squared pairwise distances.  */
  /*  */
  b_sp = 0.0;
  for (i = 0; i < 9; i++) {
    b_i = 8 - i;
    for (j = 0; j <= b_i; j++) {
      xpageoffset = (i + j) + 2;
      if (xpageoffset > 10) {
        emlrtDynamicBoundsCheckR2012b(xpageoffset, 1, 10, &c_emlrtBCI, &st);
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

  b_sp /= 45.0;

  /*  spC = separation_constrained(pos, params); */
  /*  cohesion */
  /*  asd = sum_sq_distances( pos ); */
  /*  target cost */
  /* SUM_SQ_DISTANCES_TGT Summary of this function goes here */
  for (xpageoffset = 0; xpageoffset < 20; xpageoffset++) {
    circle_tgt = pos[xpageoffset] - params_tgt[xpageoffset];
    z1[xpageoffset] = circle_tgt * circle_tgt;
  }

  for (i = 0; i < 10; i++) {
    xpageoffset = i << 1;
    y[i] = z1[xpageoffset] + z1[xpageoffset + 1];
  }

  circle_tgt = y[0];
  for (xpageoffset = 0; xpageoffset < 9; xpageoffset++) {
    circle_tgt += y[xpageoffset + 1];
  }

  /*  vm */
  /*  vm = velocity_matching(vel, params) ; */
  /*  Predator avoiadance */
  st.site = &k_emlrtRSI;
  if (muDoubleScalarIsNaN(params_predator)) {
    emlrtErrorWithMessageIdR2018a(&st, &b_emlrtRTEI, "MATLAB:nologicalnan",
      "MATLAB:nologicalnan", 0);
  }

  if (params_predator != 0.0) {
    st.site = &l_emlrtRSI;
    predator_avoidance(&st, pos, params_n, params_ws, params_wc, params_wp);
  }

  /*  res = asd + 1000000 * spC; */
  res = params_wt * circle_tgt + params_ws * b_sp;

  /*      + params.wo * obst... */
  /*      + params.wt * tgt; */
  /*  tgt weight: 1, 10, 50  */
  return res;
}

/* End of code generation (fitness.c) */
