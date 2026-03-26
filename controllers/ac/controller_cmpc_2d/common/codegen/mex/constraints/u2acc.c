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
#include "constraints.h"
#include "constraints_emxutil.h"
#include "mwmathutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo c_emlrtRSI = { 6,   /* lineNo */
  "u2acc",                             /* fcnName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/common/u2acc.m"/* pathName */
};

static emlrtRSInfo d_emlrtRSI = { 18,  /* lineNo */
  "reshapeSizeChecks",                 /* fcnName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pathName */
};

static emlrtRTEInfo b_emlrtRTEI = { 59,/* lineNo */
  23,                                  /* colNo */
  "reshapeSizeChecks",                 /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pName */
};

static emlrtRTEInfo c_emlrtRTEI = { 57,/* lineNo */
  23,                                  /* colNo */
  "reshapeSizeChecks",                 /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pName */
};

static emlrtRTEInfo d_emlrtRTEI = { 52,/* lineNo */
  13,                                  /* colNo */
  "reshapeSizeChecks",                 /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/reshapeSizeChecks.m"/* pName */
};

static emlrtRTEInfo e_emlrtRTEI = { 49,/* lineNo */
  19,                                  /* colNo */
  "assertValidSizeArg",                /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/assertValidSizeArg.m"/* pName */
};

static emlrtRTEInfo f_emlrtRTEI = { 64,/* lineNo */
  15,                                  /* colNo */
  "assertValidSizeArg",                /* fName */
  "/Applications/MATLAB_R2019b.app/toolbox/eml/eml/+coder/+internal/assertValidSizeArg.m"/* pName */
};

static emlrtRTEInfo j_emlrtRTEI = { 6, /* lineNo */
  1,                                   /* colNo */
  "u2acc",                             /* fName */
  "/Users/Usama/UmehmoodGoogle/Work/Code/BB-Simplex/common/u2acc.m"/* pName */
};

/* Function Definitions */
void u2acc(const emlrtStack *sp, const real_T u[84], real_T n, real_T h,
           emxArray_real_T *acc)
{
  real_T varargin_1[3];
  int32_T k;
  boolean_T guard1 = false;
  int32_T exitg2;
  boolean_T exitg1;
  real_T b_n;
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
  st.site = &c_emlrtRSI;
  varargin_1[0] = n;
  varargin_1[1] = 2.0;
  varargin_1[2] = h;
  b_st.site = &d_emlrtRSI;
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
    emlrtErrorWithMessageIdR2018a(&b_st, &e_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector",
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  if (n <= 0.0) {
    b_n = 0.0;
  } else {
    b_n = n;
  }

  if (h <= 0.0) {
    b_n = 0.0;
  } else {
    b_n = b_n * 2.0 * h;
  }

  if (!(b_n <= 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(&b_st, &f_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      "Coder:MATLAB:pmaxsize", 0);
  }

  k = (int32_T)n;
  num_idx_2 = (int32_T)h;
  if (k > 84) {
    emlrtErrorWithMessageIdR2018a(&st, &d_emlrtRTEI,
      "Coder:toolbox:reshape_emptyReshapeLimit",
      "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }

  if (num_idx_2 > 84) {
    emlrtErrorWithMessageIdR2018a(&st, &d_emlrtRTEI,
      "Coder:toolbox:reshape_emptyReshapeLimit",
      "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }

  if ((k < 0) || (num_idx_2 < 0)) {
    emlrtErrorWithMessageIdR2018a(&st, &c_emlrtRTEI,
      "MATLAB:checkDimCommon:nonnegativeSize",
      "MATLAB:checkDimCommon:nonnegativeSize", 0);
  }

  if ((k << 1) * num_idx_2 != 84) {
    emlrtErrorWithMessageIdR2018a(&st, &b_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel",
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  i = acc->size[0] * acc->size[1] * acc->size[2];
  acc->size[0] = k;
  acc->size[1] = 2;
  acc->size[2] = num_idx_2;
  emxEnsureCapacity_real_T(sp, acc, i, &j_emlrtRTEI);
  for (i = 0; i < 84; i++) {
    acc->data[i] = u[i];
  }
}

/* End of code generation (u2acc.c) */
