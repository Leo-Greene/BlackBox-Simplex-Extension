/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_cost_sum_bb_mex.c
 *
 * Code generation for function '_coder_cost_sum_bb_mex'
 *
 */

/* Include files */
#include "_coder_cost_sum_bb_mex.h"
#include "_coder_cost_sum_bb_api.h"
#include "cost_sum_bb.h"
#include "cost_sum_bb_data.h"
#include "cost_sum_bb_initialize.h"
#include "cost_sum_bb_terminate.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void cost_sum_bb_mexFunction(int32_T nlhs, mxArray *plhs[1],
  int32_T nrhs, const mxArray *prhs[4]);

/* Function Definitions */
void cost_sum_bb_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs, const
  mxArray *prhs[4])
{
  const mxArray *outputs[1];
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;

  /* Check for proper number of arguments. */
  if (nrhs != 4) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 4, 4,
                        11, "cost_sum_bb");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 11,
                        "cost_sum_bb");
  }

  /* Call the function. */
  cost_sum_bb_api(prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(cost_sum_bb_atexit);

  /* Module initialization. */
  cost_sum_bb_initialize();

  /* Dispatch the entry-point. */
  cost_sum_bb_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  cost_sum_bb_terminate();
}

emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_cost_sum_bb_mex.c) */
