/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * cost_sum.c
 *
 * Code generation for function 'cost_sum'
 *
 */

/* Include files */
#include "cost_sum.h"
#include "cost_sum_data.h"
#include "cost_sum_emxutil.h"
#include "dynamics.h"
#include "eml_int_forloop_overflow_check.h"
#include "fitness.h"
#include "mwmathutil.h"
#include "rt_nonfinite.h"
#include "u2acc.h"

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = { 4,     /* lineNo */
  "cost_sum",                          /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pathName */
};

static emlrtRSInfo b_emlrtRSI = { 9,   /* lineNo */
  "cost_sum",                          /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pathName */
};

static emlrtRSInfo c_emlrtRSI = { 10,  /* lineNo */
  "cost_sum",                          /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pathName */
};

static emlrtRSInfo d_emlrtRSI = { 14,  /* lineNo */
  "cost_sum",                          /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pathName */
};

static emlrtRSInfo e_emlrtRSI = { 16,  /* lineNo */
  "cost_sum",                          /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pathName */
};

static emlrtRTEInfo emlrtRTEI = { 7,   /* lineNo */
  13,                                  /* colNo */
  "cost_sum",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pName */
};

static emlrtBCInfo emlrtBCI = { -1,    /* iFirst */
  -1,                                  /* iLast */
  8,                                   /* lineNo */
  21,                                  /* colNo */
  "acc",                               /* aName */
  "cost_sum",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m",/* pName */
  0                                    /* checkKind */
};

static emlrtBCInfo b_emlrtBCI = { -1,  /* iFirst */
  -1,                                  /* iLast */
  11,                                  /* lineNo */
  21,                                  /* colNo */
  "a",                                 /* aName */
  "cost_sum",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m",/* pName */
  0                                    /* checkKind */
};

static emlrtECInfo emlrtECI = { -1,    /* nDims */
  11,                                  /* lineNo */
  17,                                  /* colNo */
  "cost_sum",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pName */
};

static emlrtRTEInfo m_emlrtRTEI = { 4, /* lineNo */
  5,                                   /* colNo */
  "cost_sum",                          /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/cost_sum.m"/* pName */
};

/* Function Definitions */
real_T cost_sum(const emlrtStack *sp, const real_T u[100], real_T pos[20],
                real_T vel[20], const struct0_T *params)
{
  real_T ret;
  emxArray_real_T *acc;
  boolean_T b;
  real_T res;
  uint8_T b_u;
  int32_T i;
  int32_T h;
  int32_T loop_ub;
  int32_T i1;
  real_T y;
  real_T x[100];
  int32_T a_size[2];
  real_T a_data[200];
  int32_T iv[1];
  int32_T iv1[1];
  int32_T a_data_tmp;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_real_T(sp, &acc, 3, &m_emlrtRTEI, true);
  b = false;

  /*  cmpc */
  st.site = &emlrtRSI;
  u2acc(&st, u, params->n, params->h, acc);
  res = muDoubleScalarRound(params->ct / params->dt);
  if (res < 256.0) {
    if (res >= 0.0) {
      b_u = (uint8_T)res;
    } else {
      b_u = 0U;
    }
  } else if (res >= 256.0) {
    b_u = MAX_uint8_T;
  } else {
    b_u = 0U;
  }

  res = 0.0;
  i = (int32_T)params->h;
  emlrtForLoopVectorCheckR2012b(1.0, 1.0, params->h, mxDOUBLE_CLASS, (int32_T)
    params->h, &emlrtRTEI, sp);
  for (h = 0; h < i; h++) {
    loop_ub = acc->size[0];
    i1 = (int32_T)(h + 1U);
    if ((i1 < 1) || (i1 > acc->size[2])) {
      emlrtDynamicBoundsCheckR2012b(i1, 1, acc->size[2], &emlrtBCI, sp);
    }

    a_size[0] = 2;
    a_size[1] = acc->size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      a_data[2 * i1] = acc->data[i1 + acc->size[0] * 2 * h];
      a_data[2 * i1 + 1] = acc->data[(i1 + acc->size[0]) + acc->size[0] * 2 * h];
    }

    st.site = &b_emlrtRSI;
    if ((1 <= b_u) && (b_u > 254)) {
      b_st.site = &h_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }

    i1 = b_u;
    for (loop_ub = 0; loop_ub < i1; loop_ub++) {
      st.site = &c_emlrtRSI;
      if (muDoubleScalarIsNaN(params->predator)) {
        emlrtErrorWithMessageIdR2018a(&st, &b_emlrtRTEI, "MATLAB:nologicalnan",
          "MATLAB:nologicalnan", 0);
      }

      if (params->predator != 0.0) {
        if (a_size[1] < 1) {
          emlrtDynamicBoundsCheckR2012b(a_size[1], 1, a_size[1], &b_emlrtBCI, sp);
        }

        if (!b) {
          iv[0] = 2;
          iv1[0] = 2;
          b = true;
        }

        emlrtSubAssignSizeCheckR2012b(&iv[0], 1, &iv1[0], 1, &emlrtECI, sp);
        a_data_tmp = 2 * (a_size[1] - 1);
        a_data[a_data_tmp] = 0.0;
        a_data[a_data_tmp + 1] = 0.0;

        /*  predator does not accelerate in the model */
        /*                  [a(:,end)] = controller_predator(pos, params); %predator's accurate model */
      }

      st.site = &d_emlrtRSI;
      dynamics(&st, pos, vel, a_data, a_size, params->n, params->dt,
               params->vmax, params->predator, params->pFactor);
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b(sp);
      }
    }

    st.site = &e_emlrtRSI;
    res += fitness(&st, pos, params->n, params->ws, params->wc, params->wt,
                   params->predator, params->wp, params->tgt);
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  emxFree_real_T(&acc);
  for (loop_ub = 0; loop_ub < 100; loop_ub++) {
    x[loop_ub] = u[loop_ub] * u[loop_ub];
  }

  y = x[0];
  for (loop_ub = 0; loop_ub < 99; loop_ub++) {
    y += x[loop_ub + 1];
  }

  ret = res / params->h + y / params->h;
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
  return ret;
}

/* End of code generation (cost_sum.c) */
