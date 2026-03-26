/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * cost_sum_bb_terminate.c
 *
 * Code generation for function 'cost_sum_bb_terminate'
 *
 */

/* Include files */
#include "cost_sum_bb_terminate.h"
#include "_coder_cost_sum_bb_mex.h"
#include "cost_sum_bb.h"
#include "cost_sum_bb_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void cost_sum_bb_atexit(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void cost_sum_bb_terminate(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (cost_sum_bb_terminate.c) */
