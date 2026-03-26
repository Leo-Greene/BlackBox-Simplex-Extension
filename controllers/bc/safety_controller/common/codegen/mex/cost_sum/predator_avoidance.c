/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * predator_avoidance.c
 *
 * Code generation for function 'predator_avoidance'
 *
 */

/* Include files */
#include "predator_avoidance.h"
#include "cost_sum.h"
#include "cost_sum_data.h"
#include "cost_sum_emxutil.h"
#include "eml_int_forloop_overflow_check.h"
#include "mwmathutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo m_emlrtRSI = { 6,   /* lineNo */
  "predator_avoidance",                /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pathName */
};

static emlrtRSInfo n_emlrtRSI = { 7,   /* lineNo */
  "predator_avoidance",                /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pathName */
};

static emlrtRSInfo o_emlrtRSI = { 11,  /* lineNo */
  "predator_avoidance",                /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pathName */
};

static emlrtRSInfo p_emlrtRSI = { 28,  /* lineNo */
  "repmat",                            /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elmat/repmat.m"/* pathName */
};

static emlrtRSInfo q_emlrtRSI = { 64,  /* lineNo */
  "repmat",                            /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elmat/repmat.m"/* pathName */
};

static emlrtRSInfo r_emlrtRSI = { 55,  /* lineNo */
  "power",                             /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/ops/power.m"/* pathName */
};

static emlrtRSInfo v_emlrtRSI = { 20,  /* lineNo */
  "sum",                               /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/sum.m"/* pathName */
};

static emlrtRSInfo ab_emlrtRSI = { 16, /* lineNo */
  "sqrt",                              /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elfun/sqrt.m"/* pathName */
};

static emlrtBCInfo d_emlrtBCI = { -1,  /* iFirst */
  -1,                                  /* iLast */
  15,                                  /* lineNo */
  21,                                  /* colNo */
  "sq_dist_to_pred",                   /* aName */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m",/* pName */
  0                                    /* checkKind */
};

static emlrtDCInfo emlrtDCI = { 11,    /* lineNo */
  73,                                  /* colNo */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m",/* pName */
  1                                    /* checkKind */
};

static emlrtBCInfo e_emlrtBCI = { 1,   /* iFirst */
  10,                                  /* iLast */
  11,                                  /* lineNo */
  73,                                  /* colNo */
  "pos",                               /* aName */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m",/* pName */
  0                                    /* checkKind */
};

static emlrtECInfo b_emlrtECI = { 2,   /* nDims */
  11,                                  /* lineNo */
  29,                                  /* colNo */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pName */
};

static emlrtRTEInfo i_emlrtRTEI = { 13,/* lineNo */
  9,                                   /* colNo */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pName */
};

static emlrtBCInfo f_emlrtBCI = { 1,   /* iFirst */
  9,                                   /* iLast */
  8,                                   /* lineNo */
  49,                                  /* colNo */
  "pos",                               /* aName */
  "separation_polynomial",             /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/separation_polynomial.m",/* pName */
  0                                    /* checkKind */
};

static emlrtBCInfo g_emlrtBCI = { 1,   /* iFirst */
  9,                                   /* iLast */
  8,                                   /* lineNo */
  47,                                  /* colNo */
  "pos",                               /* aName */
  "sum_sq_distances",                  /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/sum_sq_distances.m",/* pName */
  0                                    /* checkKind */
};

static emlrtDCInfo b_emlrtDCI = { 31,  /* lineNo */
  14,                                  /* colNo */
  "repmat",                            /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elmat/repmat.m",/* pName */
  4                                    /* checkKind */
};

static emlrtRTEInfo j_emlrtRTEI = { 13,/* lineNo */
  9,                                   /* colNo */
  "sqrt",                              /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/elfun/sqrt.m"/* pName */
};

static emlrtRTEInfo k_emlrtRTEI = { 58,/* lineNo */
  23,                                  /* colNo */
  "assertValidSizeArg",                /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/assertValidSizeArg.m"/* pName */
};

static emlrtRTEInfo o_emlrtRTEI = { 11,/* lineNo */
  29,                                  /* colNo */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pName */
};

static emlrtRTEInfo p_emlrtRTEI = { 1, /* lineNo */
  18,                                  /* colNo */
  "predator_avoidance",                /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/cost_functions/predator_avoidance.m"/* pName */
};

/* Function Definitions */
real_T predator_avoidance(const emlrtStack *sp, const real_T pos[20], real_T
  params_n, real_T params_ws, real_T params_wc, real_T params_wp)
{
  real_T res;
  real_T b_sp;
  int32_T i;
  int32_T b_i;
  int32_T j;
  int32_T ntilecols;
  real_T asd;
  int32_T ibtile;
  real_T pa;
  real_T z1_idx_0;
  emxArray_real_T *r;
  int32_T iv[2];
  real_T a_data[20];
  int32_T y_size_idx_1;
  real_T y_data[20];
  real_T b_y_data[10];
  boolean_T p;
  real_T sq_dist_to_pred_data[10];
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);

  /*  cmpc - predator_avoidance */
  /*  cohesion and separation of the flock */
  st.site = &m_emlrtRSI;

  /*  cmpc Given an pos -> 2 x n matrix, res is sum of inverse of squared pairwise distances.  */
  /*  */
  b_sp = 0.0;
  for (i = 0; i < 8; i++) {
    b_i = 7 - i;
    for (j = 0; j <= b_i; j++) {
      ntilecols = (i + j) + 2;
      if (ntilecols > 9) {
        emlrtDynamicBoundsCheckR2012b(ntilecols, 1, 9, &f_emlrtBCI, &st);
      }

      ibtile = i << 1;
      ntilecols = (ntilecols - 1) << 1;
      pa = pos[ibtile] - pos[ntilecols];
      z1_idx_0 = pa * pa;
      pa = pos[ibtile + 1] - pos[ntilecols + 1];
      b_sp += 1.0 / (z1_idx_0 + pa * pa);
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b(&st);
      }
    }

    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(&st);
    }
  }

  b_sp /= 36.0;
  st.site = &n_emlrtRSI;

  /*  cmpc Given an pos -> 2 x n matrix, res is sum of squared pairwise distances.  */
  /*  */
  asd = 0.0;
  for (i = 0; i < 8; i++) {
    b_i = 7 - i;
    for (j = 0; j <= b_i; j++) {
      ntilecols = (i + j) + 2;
      if (ntilecols > 9) {
        emlrtDynamicBoundsCheckR2012b(ntilecols, 1, 9, &g_emlrtBCI, &st);
      }

      ibtile = i << 1;
      ntilecols = (ntilecols - 1) << 1;
      pa = pos[ibtile] - pos[ntilecols];
      z1_idx_0 = pa * pa;
      pa = pos[ibtile + 1] - pos[ntilecols + 1];
      asd += z1_idx_0 + pa * pa;
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b(&st);
      }
    }

    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(&st);
    }
  }

  asd /= 36.0;

  /*  predator cost */
  if (1.0 > params_n - 1.0) {
    b_i = 0;
  } else {
    if (params_n - 1.0 != (int32_T)muDoubleScalarFloor(params_n - 1.0)) {
      emlrtIntegerCheckR2012b(params_n - 1.0, &emlrtDCI, sp);
    }

    b_i = (int32_T)(params_n - 1.0);
    if ((b_i < 1) || (b_i > 10)) {
      emlrtDynamicBoundsCheckR2012b(b_i, 1, 10, &e_emlrtBCI, sp);
    }
  }

  st.site = &o_emlrtRSI;
  b_st.site = &p_emlrtRSI;
  if ((params_n - 1.0 != muDoubleScalarFloor(params_n - 1.0)) ||
      muDoubleScalarIsInf(params_n - 1.0) || (params_n - 1.0 < -2.147483648E+9) ||
      (params_n - 1.0 > 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(&b_st, &k_emlrtRTEI,
      "Coder:MATLAB:NonIntegerInput", "Coder:MATLAB:NonIntegerInput", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  if (params_n - 1.0 <= 0.0) {
    pa = 0.0;
  } else {
    pa = params_n - 1.0;
  }

  if (!(pa <= 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(&b_st, &g_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      "Coder:MATLAB:pmaxsize", 0);
  }

  if (!(params_n - 1.0 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(params_n - 1.0, &b_emlrtDCI, &st);
  }

  emxInit_real_T(&st, &r, 2, &p_emlrtRTEI, true);
  ibtile = r->size[0] * r->size[1];
  r->size[0] = 2;
  r->size[1] = (int32_T)(params_n - 1.0);
  emxEnsureCapacity_real_T(&st, r, ibtile, &o_emlrtRTEI);
  ntilecols = (int32_T)(params_n - 1.0);
  b_st.site = &q_emlrtRSI;
  if ((1 <= (int32_T)(params_n - 1.0)) && ((int32_T)(params_n - 1.0) >
       2147483646)) {
    c_st.site = &h_emlrtRSI;
    b_check_forloop_overflow_error(&c_st);
  }

  for (j = 0; j < ntilecols; j++) {
    ibtile = j << 1;
    r->data[ibtile] = pos[18];
    r->data[ibtile + 1] = pos[19];
  }

  iv[0] = 2;
  iv[1] = b_i;
  emlrtSizeEqCheckNDR2012b(*(int32_T (*)[2])r->size, iv, &b_emlrtECI, sp);
  st.site = &o_emlrtRSI;
  i = r->size[1];
  ntilecols = r->size[1];
  for (b_i = 0; b_i < ntilecols; b_i++) {
    j = b_i << 1;
    a_data[2 * b_i] = r->data[2 * b_i] - pos[j];
    ibtile = 2 * b_i + 1;
    a_data[ibtile] = r->data[ibtile] - pos[j + 1];
  }

  emxFree_real_T(&r);
  b_st.site = &r_emlrtRSI;
  y_size_idx_1 = (int8_T)i;
  ntilecols = (int8_T)i << 1;
  for (j = 0; j < ntilecols; j++) {
    y_data[j] = a_data[j] * a_data[j];
  }

  st.site = &o_emlrtRSI;
  b_st.site = &v_emlrtRSI;
  if (y_size_idx_1 == 0) {
    y_size_idx_1 = 0;
  } else {
    for (i = 0; i < y_size_idx_1; i++) {
      ntilecols = i << 1;
      b_y_data[i] = y_data[ntilecols] + y_data[ntilecols + 1];
    }
  }

  st.site = &o_emlrtRSI;
  p = false;
  for (j = 0; j < y_size_idx_1; j++) {
    if (p || (b_y_data[j] < 0.0)) {
      p = true;
    }
  }

  if (p) {
    emlrtErrorWithMessageIdR2018a(&st, &j_emlrtRTEI,
      "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4,
      4, "sqrt");
  }

  b_st.site = &ab_emlrtRSI;
  for (j = 0; j < y_size_idx_1; j++) {
    b_y_data[j] = muDoubleScalarSqrt(b_y_data[j]);
  }

  st.site = &o_emlrtRSI;
  b_st.site = &r_emlrtRSI;
  for (j = 0; j < y_size_idx_1; j++) {
    sq_dist_to_pred_data[j] = muDoubleScalarPower(b_y_data[j], 3.0);
  }

  pa = 0.0;
  b_i = (int32_T)(params_n - 1.0);
  emlrtForLoopVectorCheckR2012b(1.0, 1.0, params_n - 1.0, mxDOUBLE_CLASS,
    (int32_T)(params_n - 1.0), &i_emlrtRTEI, sp);
  for (i = 0; i < b_i; i++) {
    /*      if sq_dist_to_pred(i) < params.pred_radius^2 */
    ibtile = (int32_T)(i + 1U);
    if ((ibtile < 1) || (ibtile > y_size_idx_1)) {
      emlrtDynamicBoundsCheckR2012b(ibtile, 1, y_size_idx_1, &d_emlrtBCI, sp);
    }

    pa += 1.0 / sq_dist_to_pred_data[ibtile - 1];

    /*      end */
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  /*  pa = sum(1./ sq_dist_to_pred); */
  res = (params_wc * asd + params_ws * b_sp) + params_wp * pa;
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
  return res;
}

/* End of code generation (predator_avoidance.c) */
