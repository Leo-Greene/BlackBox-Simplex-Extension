/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * sum.c
 *
 * Code generation for function 'sum'
 *
 */

/* Include files */
#include "sum.h"
#include "constraints.h"
#include "constraints_emxutil.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo i_emlrtRSI = { 21,  /* lineNo */
  "eml_int_forloop_overflow_check",    /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"/* pathName */
};

static emlrtRSInfo j_emlrtRSI = { 20,  /* lineNo */
  "sum",                               /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/sum.m"/* pathName */
};

static emlrtRSInfo k_emlrtRSI = { 96,  /* lineNo */
  "sumprod",                           /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/sumprod.m"/* pathName */
};

static emlrtRSInfo l_emlrtRSI = { 124, /* lineNo */
  "combineVectorElements",             /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pathName */
};

static emlrtRSInfo m_emlrtRSI = { 163, /* lineNo */
  "colMajorFlatIter",                  /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pathName */
};

static emlrtRSInfo n_emlrtRSI = { 168, /* lineNo */
  "colMajorFlatIter",                  /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pathName */
};

static emlrtRSInfo o_emlrtRSI = { 186, /* lineNo */
  "colMajorFlatIter",                  /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pathName */
};

static emlrtRTEInfo n_emlrtRTEI = { 20,/* lineNo */
  1,                                   /* colNo */
  "sum",                               /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/sum.m"/* pName */
};

static emlrtRTEInfo o_emlrtRTEI = { 124,/* lineNo */
  13,                                  /* colNo */
  "combineVectorElements",             /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pName */
};

/* Function Definitions */
void sum(const emlrtStack *sp, const emxArray_real_T *x, emxArray_real_T *y)
{
  uint8_T sz_idx_0;
  int32_T vstride;
  uint8_T sz_idx_2;
  int32_T pagesize;
  int32_T iy;
  int32_T npages;
  int32_T k;
  int32_T i;
  int32_T ypageoffset;
  int32_T j;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &j_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  e_st.prev = &d_st;
  e_st.tls = d_st.tls;
  b_st.site = &k_emlrtRSI;
  if ((x->size[0] == 0) || (x->size[2] == 0)) {
    sz_idx_0 = (uint8_T)x->size[0];
    sz_idx_2 = (uint8_T)x->size[2];
    iy = y->size[0] * y->size[1] * y->size[2];
    y->size[0] = sz_idx_0;
    y->size[1] = 1;
    y->size[2] = sz_idx_2;
    emxEnsureCapacity_real_T(&b_st, y, iy, &n_emlrtRTEI);
    k = sz_idx_0 * sz_idx_2;
    for (iy = 0; iy < k; iy++) {
      y->data[iy] = 0.0;
    }
  } else {
    c_st.site = &l_emlrtRSI;
    vstride = x->size[0] - 1;
    pagesize = x->size[0] << 1;
    npages = 1;
    k = 3;
    if (x->size[2] == 1) {
      k = 2;
    }

    for (iy = 3; iy <= k; iy++) {
      npages *= x->size[2];
    }

    iy = y->size[0] * y->size[1] * y->size[2];
    y->size[0] = (uint8_T)x->size[0];
    y->size[1] = 1;
    y->size[2] = (uint8_T)x->size[2];
    emxEnsureCapacity_real_T(&c_st, y, iy, &o_emlrtRTEI);
    d_st.site = &m_emlrtRSI;
    if ((1 <= npages) && (npages > 2147483646)) {
      e_st.site = &i_emlrtRSI;
      check_forloop_overflow_error(&e_st);
    }

    for (i = 0; i < npages; i++) {
      k = i * pagesize;
      ypageoffset = i * (vstride + 1);
      d_st.site = &n_emlrtRSI;
      for (j = 0; j <= vstride; j++) {
        y->data[ypageoffset + j] = x->data[k + j];
      }

      k += vstride;
      d_st.site = &o_emlrtRSI;
      for (j = 0; j <= vstride; j++) {
        iy = ypageoffset + j;
        y->data[iy] += x->data[(k + j) + 1];
      }
    }
  }
}

/* End of code generation (sum.c) */
