// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate.
#include <array>      // For std::array.
#include <cmath>
#include <cstddef>  // For size_t.
#include <cstdint>
#include <memory>  // For std::unique_ptr.
#include <random>  // For std::uniform_real_distribution.
#include <vector>  // For std::vector.

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"
#include "subgraph-unary-tester.h"

using HardSwishTestF16 = UnaryTest<xnn_float16>;
using HardSwishTestF32 = UnaryTest<float>;

TEST_F(HardSwishTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(0, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  const std::array<size_t, 3> dims = {{1, 3, 5}};
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_hardswish(subgraph, input_id, output_id, 0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_hardswish);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(HardSwishTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(0, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  const std::array<size_t, 3> dims = {{1, 3, 5}};
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_hardswish(subgraph, input_id, output_id, 0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_hardswish);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(HardSwishTestF16, matches_operator_api)
{
  xnnpack::Buffer<xnn_float16> input(num_output_elements + XNN_EXTRA_BYTES / sizeof(xnn_float16), std::nanf(""));
  std::uniform_real_distribution<float> f32dist(-4.0f, 4.0f);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  xnnpack::Buffer<xnn_float16> subgraph_output(num_output_elements, std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_hardswish_nc_f16(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  xnnpack::Buffer<xnn_float16> operator_output(num_output_elements, std::nanf(""));
  ASSERT_EQ(xnn_status_success, xnn_reshape_hardswish_nc_f16(op, batch_size, channels, channels, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_hardswish_nc_f16(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  ASSERT_NE(nullptr, subgraph);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id =  XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, /*external_id=*/0,
                          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, /*external_id=*/1,
                          XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(xnn_status_success, xnn_define_hardswish(subgraph, input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output_id, subgraph_output.data()}
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < num_output_elements; i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(HardSwishTestF32, matches_operator_api)
{
  xnnpack::Buffer<float> input(num_output_elements + XNN_EXTRA_BYTES / sizeof(float), std::nanf(""));
  std::uniform_real_distribution<float> f32dist(-4.0f, 4.0f);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  xnnpack::Buffer<float> subgraph_output(num_output_elements, std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_hardswish_nc_f32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  xnnpack::Buffer<float> operator_output(num_output_elements, std::nanf(""));
  ASSERT_EQ(xnn_status_success, xnn_reshape_hardswish_nc_f32(op, batch_size, channels, channels, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_hardswish_nc_f32(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  ASSERT_NE(nullptr, subgraph);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id =  XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/0,
                          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/1,
                          XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(xnn_status_success, xnn_define_hardswish(subgraph, input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output_id, subgraph_output.data()}
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < num_output_elements; i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}
