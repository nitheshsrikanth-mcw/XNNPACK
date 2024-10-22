// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: f32-vmulcaddc
//   Generator: tools/generate-vmulcaddc-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vmulcaddc.h"
#include "vmulcaddc-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile,vector_tile, datatype, params_type, init_params) XNN_TEST_VMULCADDC_ROW_EQ(ukernel,arch_flags, row_tile, vector_tile, datatype, params_type, init_params);
#include "src/f32-vmulcaddc/f32-vmulcaddc.h"
#undef XNN_UKERNEL_WITH_PARAMS
