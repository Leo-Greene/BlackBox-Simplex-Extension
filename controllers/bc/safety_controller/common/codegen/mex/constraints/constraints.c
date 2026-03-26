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

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = { 4,     /* lineNo */
  "constraints",                       /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/common/constraints.m"/* pathName */
};

static emlrtRSInfo b_emlrtRSI = { 5,   /* lineNo */
  "constraints",                       /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/common/constraints.m"/* pathName */
};

static emlrtRTEInfo emlrtRTEI = { 13,  /* lineNo */
  9,                                   /* colNo */
  "sqrt",                              /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elfun/sqrt.m"/* pName */
};

static emlrtRTEInfo h_emlrtRTEI = { 5, /* lineNo */
  1,                                   /* colNo */
  "constraints",                       /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/common/constraints.m"/* pName */
};

static emlrtRTEInfo i_emlrtRTEI = { 6, /* lineNo */
  1,                                   /* colNo */
  "constraints",                       /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/common/constraints.m"/* pName */
};

static emlrtRTEInfo j_emlrtRTEI = { 4, /* lineNo */
  1,                                   /* colNo */
  "constraints",                       /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/common/constraints.m"/* pName */
};

static emlrtRTEInfo k_emlrtRTEI = { 5, /* lineNo */
  16,                                  /* colNo */
  "constraints",                       /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/safety_controller/common/constraints.m"/* pName */
};

/* Function Definitions */
void constraints(const emlrtStack *sp, const real_T u[140], const struct0_T
                 *params, emxArray_real_T *c)
{
  emxArray_real_T *acc;
  emxArray_real_T *res;
  emxArray_real_T *r;
  boolean_T p;
  int32_T i;
  int32_T k;
  int32_T nx;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_real_T(sp, &acc, 3, &j_emlrtRTEI, true);
  emxInit_real_T(sp, &res, 3, &h_emlrtRTEI, true);
  emxInit_real_T(sp, &r, 3, &k_emlrtRTEI, true);

  /*  constraints - non-linear constraints to enforce magnitude of acceleration */
  /*  */
  st.site = &emlrtRSI;
  u2acc(&st, u, params->n, params->h_bc, acc);
  st.site = &b_emlrtRSI;
  b_st.site = &b_emlrtRSI;
  power(&b_st, acc, r);
  b_st.site = &b_emlrtRSI;
  sum(&b_st, r, res);
  p = false;
  i = res->size[0] * res->size[2];
  emxFree_real_T(&r);
  emxFree_real_T(&acc);
  for (k = 0; k < i; k++) {
    if (p || (res->data[k] < 0.0)) {
      p = true;
    }
  }

  if (p) {
    emlrtErrorWithMessageIdR2018a(&st, &emlrtRTEI,
      "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4,
      4, "sqrt");
  }

  nx = res->size[0] * res->size[2];
  for (k = 0; k < nx; k++) {
    res->data[k] = muDoubleScalarSqrt(res->data[k]);
  }

  nx = res->size[0] * res->size[1] * res->size[2];
  i = res->size[0] * res->size[1] * res->size[2];
  res->size[1] = 1;
  emxEnsureCapacity_real_T(sp, res, i, &h_emlrtRTEI);
  for (i = 0; i < nx; i++) {
    res->data[i] -= params->amax;
  }

  i = c->size[0];
  c->size[0] = res->size[0] * res->size[2];
  emxEnsureCapacity_real_T(sp, c, i, &i_emlrtRTEI);
  nx = res->size[0] * res->size[2];
  for (i = 0; i < nx; i++) {
    c->data[i] = res->data[i];
  }

  emxFree_real_T(&res);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

/* End of code generation (constraints.c) */
