/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * cost_sum_data.c
 *
 * Code generation for function 'cost_sum_data'
 *
 */

/* Include files */
#include "cost_sum_data.h"
#include "cost_sum.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
emlrtCTX emlrtRootTLSGlobal = NULL;
const volatile char_T *emlrtBreakCheckR2012bFlagVar = NULL;
emlrtContext emlrtContextGlobal = { true,/* bFirstTime */
  false,                               /* bInitialized */
  131483U,                             /* fVersionInfo */
  NULL,                                /* fErrorFunction */
  "cost_sum",                          /* fFunctionName */
  NULL,                                /* fRTCallStack */
  false,                               /* bDebugMode */
  { 2045744189U, 2170104910U, 2743257031U, 4284093946U },/* fSigWrd */
  NULL                                 /* fSigMem */
};

emlrtRSInfo h_emlrtRSI = { 21,         /* lineNo */
  "eml_int_forloop_overflow_check",    /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"/* pathName */
};

emlrtRSInfo s_emlrtRSI = { 64,         /* lineNo */
  "fltpower",                          /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/ops/power.m"/* pathName */
};

emlrtRSInfo t_emlrtRSI = { 66,         /* lineNo */
  "applyBinaryScalarFunction",         /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/applyBinaryScalarFunction.m"/* pathName */
};

emlrtRSInfo u_emlrtRSI = { 188,        /* lineNo */
  "flatIter",                          /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/applyBinaryScalarFunction.m"/* pathName */
};

emlrtRSInfo w_emlrtRSI = { 96,         /* lineNo */
  "sumprod",                           /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/sumprod.m"/* pathName */
};

emlrtRSInfo x_emlrtRSI = { 124,        /* lineNo */
  "combineVectorElements",             /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pathName */
};

emlrtRSInfo y_emlrtRSI = { 163,        /* lineNo */
  "colMajorFlatIter",                  /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/lib/matlab/datafun/private/combineVectorElements.m"/* pathName */
};

emlrtRSInfo bb_emlrtRSI = { 33,        /* lineNo */
  "applyScalarFunctionInPlace",        /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/applyScalarFunctionInPlace.m"/* pathName */
};

emlrtRTEInfo b_emlrtRTEI = { 13,       /* lineNo */
  13,                                  /* colNo */
  "toLogicalCheck",                    /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/toLogicalCheck.m"/* pName */
};

emlrtRTEInfo g_emlrtRTEI = { 64,       /* lineNo */
  15,                                  /* colNo */
  "assertValidSizeArg",                /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/assertValidSizeArg.m"/* pName */
};

/* End of code generation (cost_sum_data.c) */
