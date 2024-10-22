// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include<iostream>
#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"
#include "replicable_random_device.h"

class VMulCAddCMicrokernelTester {
 public:
  VMulCAddCMicrokernelTester& channel_tile(size_t channel_tile) {
    this->channel_tile_ = channel_tile;
    return *this;
  }

  size_t channel_tile() const {
    return this->channel_tile_;
  }

  VMulCAddCMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  size_t packed_channels() const {
    return channels() % channel_tile() == 0 ? channels() : (channels() / channel_tile() + 1) * channel_tile();
  }

  VMulCAddCMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const {
    return this->rows_;
  }

  VMulCAddCMicrokernelTester& input_stride(size_t input_stride) {
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    return this->input_stride_ == 0 ? channels() : this->input_stride_;
  }

  VMulCAddCMicrokernelTester& output_stride(size_t output_stride) {
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    return this->output_stride_ == 0 ? channels() : this->output_stride_;
  }

  VMulCAddCMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  VMulCAddCMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  VMulCAddCMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  VMulCAddCMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_vmulcaddc_ukernel_fn vmulcaddc, xnn_init_f16_minmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    if (inplace()) {
      ASSERT_EQ(input_stride(), output_stride());
    }

    std::vector<xnn_float16> x((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16> scale(channels());
    std::vector<xnn_float16> bias(channels());
    std::vector<xnn_float16, AlignedAllocator<xnn_float16, 64>> packed_w(packed_channels() * 2);
    std::vector<xnn_float16> y((rows() - 1) * output_stride() + channels() + (inplace() ? XNN_EXTRA_BYTES / sizeof(xnn_float16) : 0));
    std::vector<float> y_ref(rows() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), std::nanf(""));
      }
      const xnn_float16* x_data = inplace() ? y.data() : x.data();

      std::fill(packed_w.begin(), packed_w.end(), std::nanf(""));
      xnn_pack_f16_vmulcaddc_w(channels(), channel_tile(),
                               reinterpret_cast<const uint16_t*>(scale.data()),
                               reinterpret_cast<const uint16_t*>(bias.data()),
                               reinterpret_cast<uint16_t*>(packed_w.data()),
                               nullptr);

      // Compute reference results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          y_ref[i * channels() + j] = x_data[i * input_stride() + j] * scale[j] + bias[j];
        }
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = xnn_float16(accumulated_max - accumulated_range / 255.0f * float(255 - qmax()));
      const float y_min = xnn_float16(accumulated_min + accumulated_range / 255.0f * float(qmin()));

      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params, y_min, y_max);

      // Call optimized micro-kernel.
      vmulcaddc(rows(), channels() * sizeof(xnn_float16),
        x_data, input_stride() * sizeof(xnn_float16),
        packed_w.data(),
        y.data(), output_stride() * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          EXPECT_NEAR(y[i * output_stride() + j], y_ref[i * channels() + j], std::max(1.0e-4f, std::abs(y_ref[i * channels() + j]) * 1.0e-2f))
            << "at pixel " << i << " / " << rows()
            << ", channel = " << j << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_f32_vmulcaddc_ukernel_fn vmulcaddc, xnn_init_f32_minmax_params_fn init_params) const {
    std::cout<<"entered test"<<std::endl;
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    if (inplace()) {
      ASSERT_EQ(input_stride(), output_stride());
    }

    std::vector<float> x((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> scale(channels());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_channels() * 2);
    std::vector<float> y((rows() - 1) * output_stride() + channels() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(rows() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      std::fill(packed_w.begin(), packed_w.end(), nanf(""));
      xnn_pack_f32_vmulcaddc_w(channels(), channel_tile(),
        scale.data(), bias.data(), packed_w.data(), nullptr);
std::cout<<"before t1"<<std::endl;
      // Compute reference results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          y_ref[i * channels() + j] = x_data[i * input_stride() + j] * scale[j] + bias[j];
        }
      }
      std::cout<<"before t4"<<std::endl;

      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      const float y_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      for (float& y_value : y_ref) {
        y_value = std::max<float>(std::min<float>(y_value, y_max), y_min);
      }
std::cout<<"before t5"<<std::endl;
      // Prepare parameters.
      std::cout<<"y min"<<y_min<<std::endl;
      std::cout<<"y max"<<y_max<<std::endl;

      xnn_f32_minmax_params params;
      params.scalar.max=y_max;
      std::cout<<"params max"<<params.scalar.max<<std::endl;
      std::cout<<"init =="<<(init_params==nullptr)<<std::endl;
      init_params(&params, y_min, y_max);
      
std::cout<<"entered t2"<<std::endl;
      // Call optimized micro-kernel.
      vmulcaddc(rows(), channels() * sizeof(float),
        x_data, input_stride() * sizeof(float),
        packed_w.data(),
        y.data(), output_stride() * sizeof(float),
        &params);
std::cout<<"entered test3"<<std::endl;
      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          EXPECT_NEAR(y[i * output_stride() + j], y_ref[i * channels() + j], std::abs(y_ref[i * channels() + j]) * 1.0e-6f)
            << "at pixel " << i << " / " << rows()
            << ", channel = " << j << " / " << channels();
        }
      }
    }
  }

 private:
  size_t channel_tile_{1};
  size_t channels_{1};
  size_t rows_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};

#define XNN_TEST_VMULCADDC_ROW_EQ(ukernel,arch_flags, row_tile, vector_tile,datatype, params_type, init_params)               \
  TEST(ukernel, ROW_eq) {                                                                       \
    VMulCAddCMicrokernelTester()                                                                \
    .rows(row_tile)                                                                             \
    .Test(ukernel,init_params);                                                                             \
  }
// #define XNN_TEST_VMULCADDC_ROW_DIV(ukernel,arch_flags, row_tile, channel_tile ...)                                                 \
//   TEST(ukernel, ROW_gt) {                                                                            \
//     for (size_t ROW_size = row_tile + 1; ROW_size < 2 * row_tile; ROW_size++) {              \
//       VMulCAddCMicrokernelTester()  \
//       .channel_tile(channel_tile)                                                                   \
//         .rows(ROW_size)                                                                           \
//         .Test(ukernel);                                                                            \
//     }\
//   }
// #define XNN_TEST_VMULCADDC_ROW_LT(ukernel,arch_flags, row_tile, channel_tile ...)                                                 \
//  TEST(ukernel, ROW_lt) {                                                                             \
//     for (size_t ROW_size =  1; ROW_size < row_tile; ROW_size++) {                              \
//       VMulCAddCMicrokernelTester()    \
//       .channel_tile(channel_tile)                                                              \
//         .rows(ROW_size)                                                                             \
//         .Test(ukernel);                                                                            \
//     }\
//   }
// #define XNN_TEST_VMULCADDC_ROW_GT(ukernel,arch_flags, row_tile, channel_tile ...)                                                 
//  TEST(ukernel, ROW_div) {                                                                            
//     for (size_t ROW_size =  2 * row_tile; ROW_size < 10 * row_tile; ROW_size+= row_tile) { 
//       VMulCAddCMicrokernelTester()     
//                .channel_tile(channel_tile)                                                    
//         .rows(ROW_size)                                                                                   
//         .Test(ukernel);                                                                            
//     }
//   }