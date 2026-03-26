/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * constraints.c
 *
 * Code generation for function 'constraints'
 *
 */

/* Include files */
#include "constraints.h"
#include "constraints_emxutil.h"
#include "mwmathutil.h"
#include "power.h"
#include "rt_nonfinite.h"
#include "sum.h"
#include "u2acc.h"
#include <string.h>

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = { 4,     /* lineNo */
  "constraints",                       /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/controller_cmpc_2d/common/constraints.m"/* pathName */
};

static emlrtRSInfo b_emlrtRSI = { 5,   /* lineNo */
  "constraints",                       /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/controller_cmpc_2d/common/constraints.m"/* pathName */
};

static emlrtRTEInfo emlrtRTEI = { 13,  /* lineNo */
  9,                                   /* colNo */
  "sqrt",                              /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elfun/sqrt.m"/* pName */
};

static emlrtRTEInfo h_emlrtRTEI = { 4, /* lineNo */
  1,                                   /* colNo */
  "constraints",                       /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/controller_cmpc_2d/common/constraints.m"/* pName */
};

static emlrtRTEInfo i_emlrtRTEI = { 5, /* lineNo */
  16,                                  /* colNo */
  "constraints",                       /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/controller_cmpc_2d/common/constraints.m"/* pName */
};

/* Function Definitions */
void constraints(const emlrtStack *sp, const real_T u[84], const struct0_T
                 *params, real_T c_data[], int32_T c_size[1])
{
  emxArray_real_T *acc;
  emxArray_real_T *r;
  real_T res_data[7056];
  int32_T res_size[3];
  boolean_T p;
  int32_T loop_ub;
  int32_T k;
  int32_T i;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_real_T(sp, &acc, 3, &h_emlrtRTEI, true);
  emxInit_real_T(sp, &r, 3, &i_emlrtRTEI, true);

  /*  constraints - non-linear constraints to enforce magnitude of acceleration */
  /*  */
  st.site = &emlrtRSI;
  u2acc(&st, u, params->n, params->h_ac, acc);
  st.site = &b_emlrtRSI;
  b_st.site = &b_emlrtRSI;
  power(&b_st, acc, r);
  b_st.site = &b_emlrtRSI;
  sum(&b_st, r, res_data, res_size);
  p = false;
  loop_ub = res_size[0] * res_size[2];
  emxFree_real_T(&r);
  emxFree_real_T(&acc);
  for (k = 0; k < loop_ub; k++) {
    if (p || (res_data[k] < 0.0)) {
      p = true;
    }
  }

  if (p) {
    emlrtErrorWithMessageIdR2018a(&st, &emlrtRTEI,
      "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4,
      4, "sqrt");
  }

  for (k = 0; k < loop_ub; k++) {
    res_data[k] = muDoubleScalarSqrt(res_data[k]);
  }

  k = res_size[0] * res_size[1] * res_size[2];
  for (i = 0; i < k; i++) {
    res_data[i] -= params->amax;
  }

  c_size[0] = loop_ub;
  if (0 <= loop_ub - 1) {
    memcpy(&c_data[0], &res_data[0], loop_ub * sizeof(real_T));
  }

  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

/* End of code generation (constraints.c) */
