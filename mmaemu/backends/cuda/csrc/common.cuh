// mmaemu CUDA kernels: shared device helpers and compile-time E4M3 decode tables.
//
// Device code only: no torch headers, no CUDA runtime headers, no host code. The same text is compiled by nvcc
// (tools/build_fatbin.py, release fatbins) and by NVRTC at run time (mmaemu/backends/cuda/nvrtc.py), with identical
// arithmetic flags (--fmad=false). NVRTC ships no standard headers, so this file includes none.
//
// The decode tables are initialised at compile time (`__constant__` initialisers), so every CUDA context that loads a
// module gets them; there is no run-time upload and no process-wide "initialised" flag (the tcemu prototype uploaded
// them once per process, so a second context computed with zero tables).
//
// The numerical model the kernels implement is translated and adapted from the MATLAB Tensor Core toolbox
// (https://github.com/north-numerical-computing/MATLAB-tensor-core, BSD 2-Clause, Copyright (c) 2025 Faizan A.
// Khattak and Mantas Mikaitis); see NOTICE.
#pragma once

#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#include <stddef.h>
#endif

#define MMAEMU_KERNEL_ABI_VERSION 1

namespace mmaemu {

// ---------------------------------------------------------------------------------------------------------------
// Compile-time E4M3 tables. Byte v = s eeee mmm.
//   value: zero -> +-0; subnormal (e == 0) -> m * 2^-9; normal -> (8 + m) * 2^(e - 10);
//          NaN codes 0x7F / 0xFF -> canonical NaN 0x7FFFFFFF (NaN propagates exactly as on hardware:
//          any NaN code in row m of A or column n of B makes C[m][n] NaN, including NaN * 0).
//   exponent used for block alignment: input exponents clamped at emin = -6 (subnormals keep a denormalised
//          significand); zero and NaN codes -> ZERO_EXP8, below every product exponent (their products never set
//          the block maximum).
constexpr int ZERO_EXP8 = -60;  // fits int8, and ea + eb >= -120 stays within int8

__host__ __device__ constexpr uint32_t e4m3_mag_bits(int e, int m) {
  // fp32 bit pattern of the magnitude; subnormal m * 2^-9 = 2^(msb - 9) * (m / 2^msb)
  return (e == 0 && m == 0) ? 0u
         : (e == 15 && m == 7) ? 0x7fffffffu
         : (e != 0) ? (((uint32_t)(e + 120) << 23) | ((uint32_t)m << 20))
         : (m >= 4) ? ((uint32_t)(127 - 9 + 2) << 23 | ((uint32_t)(m - 4) << 21))
         : (m >= 2) ? ((uint32_t)(127 - 9 + 1) << 23 | ((uint32_t)(m - 2) << 22))
                    : ((uint32_t)(127 - 9) << 23);
}
__host__ __device__ constexpr uint32_t e4m3_val_bits(int v) {
  // NaN codes give 0x7FFFFFFF for both signs (canonical NaN); other negative codes set the sign bit
  return ((v & 0x7f) == 0x7f) ? 0x7fffffffu
                              : (e4m3_mag_bits((v >> 3) & 15, v & 7) | ((v & 0x80) ? 0x80000000u : 0u));
}
__host__ __device__ constexpr int e4m3_exp8(int v) {
  return ((v & 0x7f) == 0 || (v & 0x7f) == 0x7f) ? ZERO_EXP8 : (((v >> 3) & 15) == 0 ? -6 : ((v >> 3) & 15) - 7);
}

#define MMAEMU_T4(f, b) f(b), f(b + 1), f(b + 2), f(b + 3)
#define MMAEMU_T16(f, b) MMAEMU_T4(f, b), MMAEMU_T4(f, b + 4), MMAEMU_T4(f, b + 8), MMAEMU_T4(f, b + 12)
#define MMAEMU_T64(f, b) MMAEMU_T16(f, b), MMAEMU_T16(f, b + 16), MMAEMU_T16(f, b + 32), MMAEMU_T16(f, b + 48)
#define MMAEMU_T256(f) MMAEMU_T64(f, 0), MMAEMU_T64(f, 64), MMAEMU_T64(f, 128), MMAEMU_T64(f, 192)

// ---------------------------------------------------------------------------------------------------------------
// PTX wrappers (no cuda_fp16.h: NVRTC has no headers)
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
__device__ __forceinline__ float f16_bits_to_f32(uint32_t h) {  // exact (every f16 value is an fp32 value)
  float f;
  const unsigned short hs = (unsigned short)(h & 0xffffu);
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hs));
  return f;
}

}  // namespace mmaemu
