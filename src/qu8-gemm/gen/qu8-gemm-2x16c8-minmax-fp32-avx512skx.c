// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>

void xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
  const __m128i vshuffle_control_mask = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);

  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    w = (const int32_t*) w + 16;

    size_t k = kc;

    while (k >= 8 * sizeof(uint8_t)) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;

      const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));

      w = (const uint8_t*) w + 128;
      k -= 8 * sizeof(uint8_t);
    }

    // Add 4 adjacent sums
    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));

    const __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    const __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));

    const __m512i vidx = _mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0);
    __m512i vacc0x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc0x084C195D2A6E3B7F);
    __m512i vacc1x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc1x084C195D2A6E3B7F);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);

    __m256i vacc0x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0x0123456789ABCDEF, 1));
    __m256i vacc1x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc1x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc1x0123456789ABCDEF, 1));

    vacc0x0123456789AB4567CDEF = _mm256_adds_epi16(vacc0x0123456789AB4567CDEF, voutput_zero_point);
    vacc1x0123456789AB4567CDEF = _mm256_adds_epi16(vacc1x0123456789AB4567CDEF, voutput_zero_point);

    const __m128i vout0x0123456789AB4567CDEF = _mm_packus_epi16(_mm256_castsi256_si128(vacc0x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc0x0123456789AB4567CDEF, 1));
    const __m128i vout1x0123456789AB4567CDEF = _mm_packus_epi16(_mm256_castsi256_si128(vacc1x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc1x0123456789AB4567CDEF, 1));

    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x0123456789AB4567CDEF, vshuffle_control_mask);
    __m128i vout1x0123456789ABCDEF = _mm_shuffle_epi8(vout1x0123456789AB4567CDEF, vshuffle_control_mask);

    vout0x0123456789ABCDEF = _mm_max_epu8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epu8(vout1x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) c1, vout1x0123456789ABCDEF);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
