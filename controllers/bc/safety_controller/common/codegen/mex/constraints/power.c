/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * power.c
 *
 * Code generation for function 'power'
 *
 */

/* Include files */
#include "power.h"
#include "constraints.h"
#include "constraints_emxutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo e_emlrtRSI = { 55,  /* lineNo */
  "power",                             /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/ops/power.m"/* pathName */
};

static emlrtRSInfo f_emlrtRSI = { 64,  /* lineNo */
  "fltpower",                          /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/ops/power.m"/* pathName */
};

static emlrtRTEInfo m_emlrtRTEI = { 64,/* lineNo */
  5,                                   /* colNo */
  "power",                             /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/ops/power.m"/* pName */
};

/* Function Definitions */
void power(const emlrtStack *sp, const emxArray_real_T *a, emxArray_real_T *y)
{
  uint8_T unnamed_idx_0;
  uint8_T unnamed_idx_2;
  int32_T nx;
  int32_T k;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &e_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  b_st.site = &f_emlrtRSI;
  unnamed_idx_0 = (uint8_T)a->size[0];
  unnamed_idx_2 = (uint8_T)a->size[2];
  nx = y->size[0] * y->size[1] * y->size[2];
  y->size[0] = unnamed_idx_0;
  y->size[1] = 2;
  y->size[2] = unnamed_idx_2;
  emxEnsureCapacity_real_T(&b_st, y, nx, &m_emlrtRTEI);
  nx = (unnamed_idx_0 << 1) * unnamed_idx_2;
  for (k = 0; k < nx; k++) {
    y->data[k] = a->data[k] * a->data[k];
  }
}

/* End of code generation (power.c) */
