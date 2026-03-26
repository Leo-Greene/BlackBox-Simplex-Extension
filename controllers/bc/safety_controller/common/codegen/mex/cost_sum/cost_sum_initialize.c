/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * cost_sum_initialize.c
 *
 * Code generation for function 'cost_sum_initialize'
 *
 */

/* Include files */
#include "cost_sum_initialize.h"
#include "_coder_cost_sum_mex.h"
#include "cost_sum.h"
#include "cost_sum_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void cost_sum_initialize(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2012b();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, 0);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

/* End of code generation (cost_sum_initialize.c) */
