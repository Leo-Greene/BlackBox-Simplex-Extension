/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fitness_bb.c
 *
 * Code generation for function 'fitness_bb'
 *
 */

/* Include files */
#include "fitness_bb.h"
#include "blas.h"
#include "cost_sum_bb.h"
#include "cost_sum_bb_data.h"
#include "mwmathutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo i_emlrtRSI = { 5,   /* lineNo */
  "fitness_bb",                        /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/fitness_bb.m"/* pathName */
};

static emlrtRSInfo j_emlrtRSI = { 8,   /* lineNo */
  "fitness_bb",                        /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/fitness_bb.m"/* pathName */
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

static emlrtRTEInfo h_emlrtRTEI = { 3, /* lineNo */
  13,                                  /* colNo */
  "orientation_single",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/cost_functions/orientation_single.m"/* pName */
};

static emlrtBCInfo c_emlrtBCI = { 1,   /* iFirst */
  7,                                   /* iLast */
  4,                                   /* lineNo */
  20,                                  /* colNo */
  "vel",                               /* aName */
  "orientation_single",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/cost_functions/orientation_single.m",/* pName */
  0                                    /* checkKind */
};

/* Function Definitions */
real_T fitness_bb(const emlrtStack *sp, const real_T pos[14], const real_T vel
                  [14], real_T params_n, real_T params_ws_bb, real_T
                  params_w_orient)
{
  real_T b_sp;
  int32_T i;
  int32_T b_i;
  int32_T j;
  int32_T c_i;
  real_T orient_s;
  int32_T i1;
  int32_T a_tmp;
  real_T scale;
  real_T z1_idx_0;
  real_T a_data[2];
  real_T b_data[2];
  ptrdiff_t n_t;
  ptrdiff_t incx_t;
  ptrdiff_t incy_t;
  real_T absxk;
  real_T t;
  real_T y;
  real_T b_y;
  emlrtStack st;
  st.prev = sp;
  st.tls = sp->tls;

  /*  cmpc - fitness  */
  st.site = &i_emlrtRSI;

  /*  cmpc Given an pos -> 2 x n matrix, res is sum of inverse of squared pairwise distances.  */
  /*  */
  b_sp = 0.0;
  for (i = 0; i < 6; i++) {
    b_i = 5 - i;
    for (j = 0; j <= b_i; j++) {
      c_i = (i + j) + 2;
      if (c_i > 7) {
        emlrtDynamicBoundsCheckR2012b(c_i, 1, 7, &b_emlrtBCI, &st);
      }

      i1 = i << 1;
      a_tmp = (c_i - 1) << 1;
      scale = pos[i1] - pos[a_tmp];
      z1_idx_0 = scale * scale;
      scale = pos[i1 + 1] - pos[a_tmp + 1];
      b_sp += 1.0 / (z1_idx_0 + scale * scale);
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b(&st);
      }
    }

    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(&st);
    }
  }

  b_sp /= 21.0;

  /*  safety_circle = circle_boundary(pos, params); */
  /*  orient_d = orientation_diff(vel, params); */
  st.site = &j_emlrtRSI;
  orient_s = 0.0;
  b_i = (int32_T)params_n;
  emlrtForLoopVectorCheckR2012b(1.0, 1.0, params_n, mxDOUBLE_CLASS, (int32_T)
    params_n, &h_emlrtRTEI, &st);
  for (i = 0; i < b_i; i++) {
    i1 = i + 1;
    if ((i1 < 1) || (i1 > 7)) {
      emlrtDynamicBoundsCheckR2012b(i1, 1, 7, &c_emlrtBCI, &st);
    }

    c_i = i << 1;
    a_data[0] = vel[c_i];
    b_data[0] = pos[c_i];
    a_tmp = c_i + 1;
    a_data[1] = vel[a_tmp];
    b_data[1] = pos[a_tmp];
    n_t = (ptrdiff_t)2;
    incx_t = (ptrdiff_t)1;
    incy_t = (ptrdiff_t)1;
    z1_idx_0 = ddot(&n_t, &a_data[0], &incx_t, &b_data[0], &incy_t);
    scale = 3.3121686421112381E-170;
    absxk = muDoubleScalarAbs(vel[c_i]);
    if (absxk > 3.3121686421112381E-170) {
      y = 1.0;
      scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      y = t * t;
    }

    absxk = muDoubleScalarAbs(vel[a_tmp]);
    if (absxk > scale) {
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      t = absxk / scale;
      y += t * t;
    }

    y = scale * muDoubleScalarSqrt(y);
    scale = 3.3121686421112381E-170;
    absxk = muDoubleScalarAbs(pos[c_i]);
    if (absxk > 3.3121686421112381E-170) {
      b_y = 1.0;
      scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      b_y = t * t;
    }

    absxk = muDoubleScalarAbs(pos[a_tmp]);
    if (absxk > scale) {
      t = scale / absxk;
      b_y = b_y * t * t + 1.0;
      scale = absxk;
    } else {
      t = absxk / scale;
      b_y += t * t;
    }

    b_y = scale * muDoubleScalarSqrt(b_y);
    scale = 1.0 - z1_idx_0 / (y * b_y);
    orient_s += scale * scale;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(&st);
    }
  }

  return params_ws_bb * b_sp + params_w_orient * orient_s;
}

/* End of code generation (fitness_bb.c) */
