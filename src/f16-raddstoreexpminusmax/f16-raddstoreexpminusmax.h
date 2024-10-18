// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, element_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, element_tile,  datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile,  datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16, 16, xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16_acc2, 16,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32, 32,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2,xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc2 , 32,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc4, 32,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40, 40,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc2, 40,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc5, 40,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48, 48,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc2, 48,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc3, 48,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64, 64,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc2, 64, xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc4, 64,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72, 72,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2,xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72_acc3 , 72,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80, 80,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc2, 80,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc5, 80,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc2, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc3, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc6, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32, 32, xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40, 40,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2, 32,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM,xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc4 , 32,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc2, 40,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc5, 40,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48, 48,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc2, 48,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc3, 48,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64, 64,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc2, 64,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc4, 64,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72, 72, xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72_acc3, 72,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80, 80,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM,xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc2 , 80,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc5, 80,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc2, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc3, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc6, 96,  xnn_float16, struct xnn_f16_default_params,  NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif