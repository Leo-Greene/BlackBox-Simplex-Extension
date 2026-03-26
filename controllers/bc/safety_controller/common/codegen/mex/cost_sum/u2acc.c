/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * u2acc.c
 *
 * Code generation for function 'u2acc'
 *
 */

/* Include files */
#include "u2acc.h"
#include "cost_sum.h"
#include "cost_sum_data.h"
#include "cost_sum_emxutil.h"
#include "mwmathutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo f_emlrtRSI = { 6,   /* lineNo */
  "u2acc",                             /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/u2acc.m"/* pathName */
};

static emlrtRSInfo g_emlrtRSI = { 18,  /* lineNo */
  "reshapeSizeChecks",                 /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pathName */
};

static emlrtRTEInfo c_emlrtRTEI = { 59,/* lineNo */
  23,                                  /* colNo */
  "reshapeSizeChecks",                 /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pName */
};

static emlrtRTEInfo d_emlrtRTEI = { 57,/* lineNo */
  23,                                  /* colNo */
  "reshapeSizeChecks",                 /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pName */
};

static emlrtRTEInfo e_emlrtRTEI = { 52,/* lineNo */
  13,                                  /* colNo */
  "reshapeSizeChecks",                 /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pName */
};

static emlrtRTEInfo f_emlrtRTEI = { 49,/* lineNo */
  19,                                  /* colNo */
  "assertValidSizeArg",                /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/assertValidSizeArg.m"/* pName */
};

static emlrtRTEInfo n_emlrtRTEI = { 6, /* lineNo */
  1,                                   /* colNo */
  "u2acc",                             /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/Swarms New/MPCQP/controller_cmpc_2d/common/u2acc.m"/* pName */
};

/* Function Definitions */
void u2acc(const emlrtStack *sp, const real_T u[100], real_T params_n, real_T
           params_h, emxArray_real_T *acc)
{
  real_T varargin_1[3];
  int32_T k;
  boolean_T guard1 = false;
  int32_T exitg2;
  boolean_T exitg1;
  real_T b_params_n;
  int32_T num_idx_2;
  int32_T i;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;

  /* u2acc - convert column vector u to acc matric */
  /*  Output: */
  /*    - acc            % n x 2 x h - The sequence of control action over the horizon */
  /*  */
  st.site = &f_emlrtRSI;
  varargin_1[0] = params_n;
  varargin_1[1] = 2.0;
  varargin_1[2] = params_h;
  b_st.site = &g_emlrtRSI;
  k = 0;
  guard1 = false;
  do {
    exitg2 = 0;
    if (k < 3) {
      if ((varargin_1[k] != muDoubleScalarFloor(varargin_1[k])) ||
          muDoubleScalarIsInf(varargin_1[k])) {
        guard1 = true;
        exitg2 = 1;
      } else {
        k++;
        guard1 = false;
      }
    } else {
      k = 0;
      exitg2 = 2;
    }
  } while (exitg2 == 0);

  if (exitg2 != 1) {
    exitg1 = false;
    while ((!exitg1) && (k < 3)) {
      if ((varargin_1[k] < -2.147483648E+9) || (varargin_1[k] > 2.147483647E+9))
      {
        guard1 = true;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }

  if (guard1) {
    emlrtErrorWithMessageIdR2018a(&b_st, &f_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector",
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  if (params_n <= 0.0) {
    b_params_n = 0.0;
  } else {
    b_params_n = params_n;
  }

  if (params_h <= 0.0) {
    b_params_n = 0.0;
  } else {
    b_params_n = b_params_n * 2.0 * params_h;
  }

  if (!(b_params_n <= 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(&b_st, &g_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      "Coder:MATLAB:pmaxsize", 0);
  }

  k = (int32_T)params_n;
  num_idx_2 = (int32_T)params_h;
  if (k > 100) {
    emlrtErrorWithMessageIdR2018a(&st, &e_emlrtRTEI,
      "Coder:toolbox:reshape_emptyReshapeLimit",
      "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }

  if (num_idx_2 > 100) {
    emlrtErrorWithMessageIdR2018a(&st, &e_emlrtRTEI,
      "Coder:toolbox:reshape_emptyReshapeLimit",
      "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }

  if ((k < 0) || (num_idx_2 < 0)) {
    emlrtErrorWithMessageIdR2018a(&st, &d_emlrtRTEI,
      "MATLAB:checkDimCommon:nonnegativeSize",
      "MATLAB:checkDimCommon:nonnegativeSize", 0);
  }

  if ((k << 1) * num_idx_2 != 100) {
    emlrtErrorWithMessageIdR2018a(&st, &c_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel",
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  i = acc->size[0] * acc->size[1] * acc->size[2];
  acc->size[0] = k;
  acc->size[1] = 2;
  acc->size[2] = num_idx_2;
  emxEnsureCapacity_real_T(sp, acc, i, &n_emlrtRTEI);
  for (i = 0; i < 100; i++) {
    acc->data[i] = u[i];
  }
}

/* End of code generation (u2acc.c) */
