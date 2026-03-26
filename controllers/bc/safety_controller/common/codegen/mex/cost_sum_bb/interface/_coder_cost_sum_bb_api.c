/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_cost_sum_bb_api.c
 *
 * Code generation for function '_coder_cost_sum_bb_api'
 *
 */

/* Include files */
#include "_coder_cost_sum_bb_api.h"
#include "cost_sum_bb.h"
#include "cost_sum_bb_data.h"
#include "rt_nonfinite.h"
#include <string.h>

/* Function Declarations */
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[140];
static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *pos,
  const char_T *identifier))[14];
static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[14];
static void e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *params,
  const char_T *identifier, struct0_T *y);
static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  char_T *identifier))[140];
static const mxArray *emlrt_marshallOut(const real_T u);
static void f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, struct0_T *y);
static real_T g_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId);
static void h_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, real_T y[14]);
static real_T (*i_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[140];
static real_T (*j_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[14];
static real_T k_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId);
static void l_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId, real_T ret[14]);

/* Function Definitions */
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[140]
{
  real_T (*y)[140];
  y = i_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
  static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *pos,
  const char_T *identifier))[14]
{
  real_T (*y)[14];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = d_emlrt_marshallIn(sp, emlrtAlias(pos), &thisId);
  emlrtDestroyArray(&pos);
  return y;
}

static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[14]
{
  real_T (*y)[14];
  y = j_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
  static void e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *params,
  const char_T *identifier, struct0_T *y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  f_emlrt_marshallIn(sp, emlrtAlias(params), &thisId, y);
  emlrtDestroyArray(&params);
}

static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  char_T *identifier))[140]
{
  real_T (*y)[140];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(sp, emlrtAlias(u), &thisId);
  emlrtDestroyArray(&u);
  return y;
}
  static const mxArray *emlrt_marshallOut(const real_T u)
{
  const mxArray *y;
  const mxArray *m;
  y = NULL;
  m = emlrtCreateDoubleScalar(u);
  emlrtAssign(&y, m);
  return y;
}

static void f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, struct0_T *y)
{
  emlrtMsgIdentifier thisId;
  static const char * fieldNames[20] = { "n", "dt", "ct", "h_ac", "h_bc",
    "steps", "amax", "vmax", "dmin", "diameter", "switch_step", "ws", "wt",
    "ws_bb", "w_orient", "predator", "pFactor", "pred_radius", "wp", "tgt" };

  static const int32_T dims = 0;
  thisId.fParent = parentId;
  thisId.bParentIsCell = false;
  emlrtCheckStructR2012b(sp, parentId, u, 20, fieldNames, 0U, &dims);
  thisId.fIdentifier = "n";
  y->n = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 0, "n")),
    &thisId);
  thisId.fIdentifier = "dt";
  y->dt = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 1,
    "dt")), &thisId);
  thisId.fIdentifier = "ct";
  y->ct = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 2,
    "ct")), &thisId);
  thisId.fIdentifier = "h_ac";
  y->h_ac = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 3,
    "h_ac")), &thisId);
  thisId.fIdentifier = "h_bc";
  y->h_bc = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 4,
    "h_bc")), &thisId);
  thisId.fIdentifier = "steps";
  y->steps = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 5,
    "steps")), &thisId);
  thisId.fIdentifier = "amax";
  y->amax = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 6,
    "amax")), &thisId);
  thisId.fIdentifier = "vmax";
  y->vmax = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 7,
    "vmax")), &thisId);
  thisId.fIdentifier = "dmin";
  y->dmin = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 8,
    "dmin")), &thisId);
  thisId.fIdentifier = "diameter";
  y->diameter = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0,
    9, "diameter")), &thisId);
  thisId.fIdentifier = "switch_step";
  y->switch_step = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u,
    0, 10, "switch_step")), &thisId);
  thisId.fIdentifier = "ws";
  y->ws = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 11,
    "ws")), &thisId);
  thisId.fIdentifier = "wt";
  y->wt = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 12,
    "wt")), &thisId);
  thisId.fIdentifier = "ws_bb";
  y->ws_bb = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 13,
    "ws_bb")), &thisId);
  thisId.fIdentifier = "w_orient";
  y->w_orient = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0,
    14, "w_orient")), &thisId);
  thisId.fIdentifier = "predator";
  y->predator = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0,
    15, "predator")), &thisId);
  thisId.fIdentifier = "pFactor";
  y->pFactor = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0,
    16, "pFactor")), &thisId);
  thisId.fIdentifier = "pred_radius";
  y->pred_radius = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u,
    0, 17, "pred_radius")), &thisId);
  thisId.fIdentifier = "wp";
  y->wp = g_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 18,
    "wp")), &thisId);
  thisId.fIdentifier = "tgt";
  h_emlrt_marshallIn(sp, emlrtAlias(emlrtGetFieldR2017b(sp, u, 0, 19, "tgt")),
                     &thisId, y->tgt);
  emlrtDestroyArray(&u);
}

static real_T g_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId)
{
  real_T y;
  y = k_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static void h_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, real_T y[14])
{
  l_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static real_T (*i_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[140]
{
  real_T (*ret)[140];
  static const int32_T dims[1] = { 140 };

  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 1U, dims);
  ret = (real_T (*)[140])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}
  static real_T (*j_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[14]
{
  real_T (*ret)[14];
  static const int32_T dims[2] = { 2, 7 };

  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 2U, dims);
  ret = (real_T (*)[14])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T k_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId)
{
  real_T ret;
  static const int32_T dims = 0;
  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 0U, &dims);
  ret = *(real_T *)emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static void l_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId, real_T ret[14])
{
  static const int32_T dims[2] = { 2, 7 };

  real_T (*r)[14];
  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 2U, dims);
  r = (real_T (*)[14])emlrtMxGetData(src);
  memcpy(&ret[0], &(*r)[0], 14U * sizeof(real_T));
  emlrtDestroyArray(&src);
}

void cost_sum_bb_api(const mxArray * const prhs[4], int32_T nlhs, const mxArray *
                     plhs[1])
{
  const mxArray *prhs_copy_idx_1;
  const mxArray *prhs_copy_idx_2;
  real_T (*u)[140];
  real_T (*pos)[14];
  real_T (*vel)[14];
  struct0_T params;
  real_T ret;
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  (void)nlhs;
  st.tls = emlrtRootTLSGlobal;
  prhs_copy_idx_1 = emlrtProtectR2012b(prhs[1], 1, false, -1);
  prhs_copy_idx_2 = emlrtProtectR2012b(prhs[2], 2, false, -1);

  /* Marshall function inputs */
  u = emlrt_marshallIn(&st, emlrtAlias(prhs[0]), "u");
  pos = c_emlrt_marshallIn(&st, emlrtAlias(prhs_copy_idx_1), "pos");
  vel = c_emlrt_marshallIn(&st, emlrtAlias(prhs_copy_idx_2), "vel");
  e_emlrt_marshallIn(&st, emlrtAliasP(prhs[3]), "params", &params);

  /* Invoke the target function */
  ret = cost_sum_bb(&st, *u, *pos, *vel, &params);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(ret);
}

/* End of code generation (_coder_cost_sum_bb_api.c) */
