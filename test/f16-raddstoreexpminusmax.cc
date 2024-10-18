// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-raddstoreexpminusmax.yaml
//   Generator: tools/generate-raddstoreexpminusmax-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "raddstoreexpminusmax-microkernel-tester.h"


#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) XNN_TEST_RADDSTOREEXPMINUSMAX_ELEMENT_EQ(ukernel,arch_flags, element_tile, init_params);  \
XNN_TEST_RADDSTOREEXPMINUSMAX_ELEMENT_DIV(ukernel,arch_flags,  element_tile, init_params);                                                                                                       \
XNN_TEST_RADDSTOREEXPMINUSMAX_ELEMENT_LT(ukernel,arch_flags,  element_tile, init_params);                                                                                                        \
XNN_TEST_RADDSTOREEXPMINUSMAX_ELEMENT_GT(ukernel,arch_flags,  element_tile, init_params);
#include "src/f16-raddstoreexpminusmax/f16-raddstoreexpminusmax.h"
#undef XNN_UKERNEL_WITH_PARAMS
