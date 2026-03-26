/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * dynamics.c
 *
 * Code generation for function 'dynamics'
 *
 */

/* Include files */
#include "dynamics.h"
#include "cost_sum.h"
#include "cost_sum_data.h"
#include "mwmathutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo i_emlrtRSI = { 5,   /* lineNo */
  "dynamics",                          /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/dynamics.m"/* pathName */
};

static emlrtBCInfo h_emlrtBCI = { 1,   /* iFirst */
  10,                                  /* iLast */
  11,                                  /* lineNo */
  19,                                  /* colNo */
  "vel",                               /* aName */
  "dynamics",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/dynamics.m",/* pName */
  0                                    /* checkKind */
};

static emlrtBCInfo i_emlrtBCI = { 1,   /* iFirst */
  10,                                  /* iLast */
  6,                                   /* lineNo */
  22,                                  /* colNo */
  "vel",                               /* aName */
  "dynamics",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/dynamics.m",/* pName */
  0                                    /* checkKind */
};

static emlrtRTEInfo l_emlrtRTEI = { 4, /* lineNo */
  9,                                   /* colNo */
  "dynamics",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/dynamics.m"/* pName */
};

static emlrtECInfo c_emlrtECI = { 2,   /* nDims */
  3,                                   /* lineNo */
  7,                                   /* colNo */
  "dynamics",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/dynamics.m"/* pName */
};

/* Function Definitions */
void dynamics(const emlrtStack *sp, real_T pos[20], real_T vel[20], const real_T
              acc_data[], const int32_T acc_size[2], real_T params_n, real_T
              params_dt, real_T params_vmax, real_T params_predator, real_T
              params_pFactor)
{
  int32_T tmp_size[2];
  int32_T loop_ub;
  int32_T i;
  int32_T iv[2];
  real_T tmp_data[200];
  int32_T j;
  real_T scale;
  real_T absxk_tmp;
  real_T t;
  real_T y;
  int32_T b_absxk_tmp;
  real_T c_absxk_tmp;
  real_T d;
  emlrtStack st;
  st.prev = sp;
  st.tls = sp->tls;

  /*  dynamics() - cmpc */
  tmp_size[0] = 2;
  tmp_size[1] = acc_size[1];
  loop_ub = acc_size[0] * acc_size[1];
  for (i = 0; i < loop_ub; i++) {
    tmp_data[i] = params_dt * acc_data[i];
  }

  iv[0] = 2;
  iv[1] = 10;
  emlrtSizeEqCheckNDR2012b(iv, tmp_size, &c_emlrtECI, sp);
  for (i = 0; i < 20; i++) {
    vel[i] += tmp_data[i];
  }

  i = (int32_T)params_n;
  emlrtForLoopVectorCheckR2012b(1.0, 1.0, params_n, mxDOUBLE_CLASS, (int32_T)
    params_n, &l_emlrtRTEI, sp);
  for (j = 0; j < i; j++) {
    st.site = &i_emlrtRSI;
    if (muDoubleScalarIsNaN(params_predator)) {
      emlrtErrorWithMessageIdR2018a(&st, &b_emlrtRTEI, "MATLAB:nologicalnan",
        "MATLAB:nologicalnan", 0);
    }

    if ((params_predator != 0.0) && ((real_T)j + 1.0 == params_n)) {
      loop_ub = j + 1;
      if ((loop_ub < 1) || (loop_ub > 10)) {
        emlrtDynamicBoundsCheckR2012b(loop_ub, 1, 10, &i_emlrtBCI, sp);
      }

      scale = 3.3121686421112381E-170;
      loop_ub = j << 1;
      absxk_tmp = muDoubleScalarAbs(vel[loop_ub]);
      if (absxk_tmp > 3.3121686421112381E-170) {
        y = 1.0;
        scale = absxk_tmp;
      } else {
        t = absxk_tmp / 3.3121686421112381E-170;
        y = t * t;
      }

      b_absxk_tmp = (j << 1) + 1;
      c_absxk_tmp = muDoubleScalarAbs(vel[b_absxk_tmp]);
      if (c_absxk_tmp > scale) {
        t = scale / c_absxk_tmp;
        y = y * t * t + 1.0;
        scale = c_absxk_tmp;
      } else {
        t = c_absxk_tmp / scale;
        y += t * t;
      }

      y = scale * muDoubleScalarSqrt(y);
      d = params_pFactor * params_vmax;
      if (y > d) {
        scale = 3.3121686421112381E-170;
        if (absxk_tmp > 3.3121686421112381E-170) {
          y = 1.0;
          scale = absxk_tmp;
        } else {
          t = absxk_tmp / 3.3121686421112381E-170;
          y = t * t;
        }

        if (c_absxk_tmp > scale) {
          t = scale / c_absxk_tmp;
          y = y * t * t + 1.0;
          scale = c_absxk_tmp;
        } else {
          t = c_absxk_tmp / scale;
          y += t * t;
        }

        y = scale * muDoubleScalarSqrt(y);
        scale = d / y;
        vel[loop_ub] *= scale;
        vel[b_absxk_tmp] *= scale;
      }
    } else {
      loop_ub = j + 1;
      if ((loop_ub < 1) || (loop_ub > 10)) {
        emlrtDynamicBoundsCheckR2012b(loop_ub, 1, 10, &h_emlrtBCI, sp);
      }

      scale = 3.3121686421112381E-170;
      loop_ub = j << 1;
      absxk_tmp = muDoubleScalarAbs(vel[loop_ub]);
      if (absxk_tmp > 3.3121686421112381E-170) {
        y = 1.0;
        scale = absxk_tmp;
      } else {
        t = absxk_tmp / 3.3121686421112381E-170;
        y = t * t;
      }

      b_absxk_tmp = loop_ub + 1;
      c_absxk_tmp = muDoubleScalarAbs(vel[b_absxk_tmp]);
      if (c_absxk_tmp > scale) {
        t = scale / c_absxk_tmp;
        y = y * t * t + 1.0;
        scale = c_absxk_tmp;
      } else {
        t = c_absxk_tmp / scale;
        y += t * t;
      }

      y = scale * muDoubleScalarSqrt(y);
      if (y > params_vmax) {
        scale = 3.3121686421112381E-170;
        if (absxk_tmp > 3.3121686421112381E-170) {
          y = 1.0;
          scale = absxk_tmp;
        } else {
          t = absxk_tmp / 3.3121686421112381E-170;
          y = t * t;
        }

        if (c_absxk_tmp > scale) {
          t = scale / c_absxk_tmp;
          y = y * t * t + 1.0;
          scale = c_absxk_tmp;
        } else {
          t = c_absxk_tmp / scale;
          y += t * t;
        }

        y = scale * muDoubleScalarSqrt(y);
        scale = params_vmax / y;
        vel[loop_ub] *= scale;
        vel[b_absxk_tmp] *= scale;
      }
    }

    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  for (i = 0; i < 20; i++) {
    pos[i] += params_dt * vel[i];
  }
}

/* End of code generation (dynamics.c) */
