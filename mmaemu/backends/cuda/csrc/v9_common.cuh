// mmaemu v9_common.cuh -- device primitives of the v9 kernel (v9.cu) and of the E-pass probe (probes.cu).
//
// Provenance: copy of the tcemu audit's p2-int-mixed/v2_common.cuh (sha256 e44ff8b9...dfa277), the header
// whose primitives the p2 audit tested exhaustively (test_prims.py / test_prims8.py).  Changes, none numerical:
//   - self-contained for NVRTC: no system headers (stdint types are typedef'd under __CUDACC_RTC__, cuda_fp16.h is
//     replaced by the identical inline PTX of __half22float2: two cvt.f32.f16);
//   - only the primitives used by v9 cfg 1 (OPT 2) and the E-pass probe are kept (bfma2, the base-32 E-pass helpers,
//     the v2/v3 block_init / block_finish / pbits_* variants used by other configs are dropped; v9 cfg 1 has its own
//     block_init_m / block_finish_m in v9.cu);
//   - namespace p2 -> mmaemu_v9.
// Every function is device-only; the file contains no host code.  It is part of the BSD-2 derived kernels named in
// NOTICE (the Ada block-FMA model of Khattak & Mikaitis, arXiv:2512.07004).
#pragma once

#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
#else
#include <stdint.h>
#endif

namespace mmaemu_v9 {

constexpr uint32_t LA_ZERO = 60;  // LA/LB byte code of a zero operand (valid for block exponents E <= 55)

__device__ __forceinline__ uint32_t dp4a_u(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t lop3_corr(uint32_t v, uint32_t sa, uint32_t sb) {  // ~v & (sa ^ sb)
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x06;" : "=r"(d) : "r"(v), "r"(sa), "r"(sb));
  return d;
}

// E4M3 byte decode, value only (the portable path of decode_word; any byte value is accepted: the NaN codes
// 0x7F / 0xFF decode to +-480, which never reach the kernel because the Python wrapper sanitises non-finite codes).
__device__ __forceinline__ float decode_value(uint32_t byte) {
  const uint32_t e = (byte >> 3) & 15u, m = byte & 7u;
  const float mag = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
  return (byte & 0x80u) ? -mag : mag;
}

// After the last block: truncate V to 14 significant bits (13 fraction bits relative to its exponent).
__device__ __forceinline__ float final_trunc(float V) {
  const float m2 = __uint_as_float((__float_as_uint(V) & 0xff800000u) + (10u << 23));
  return __fsub_rn(__fadd_rz(V, m2), m2);
}

// ------------------------------------------------------------------------------------------------------------------
// Word-level (4 E4M3 bytes = 4 consecutive k) SIMD decode and the base-64 bf16 E-pass operands built by byte
// placement (PRMT): a bf16 half with bits 8-15 = x (x < 128), bits 0-7 = 0 is 2^(2x - 127).  With x = 3*ea + 60
// (ea = clamped exponent) a product of two operands is 2^(6*(ea+eb) - 14); <= 17 terms (16 + init) keep the sum below
// 18 * 64^E, so the exponent field is 6E + 113 + d with d in [0, 4] (the "window property", checked per device by the
// "window" gate of mmaemu/backends/cuda/fast/gates.py).

// fp32 bits of 2^E_prod (mantissa bits = garbage, exponent exact) from the base-64 field f (0 <= f < 256):
// E + 12 = floor((f - 41) / 6) = ((f - 41) * 171) >> 10;  ((f*171 + 110749) << 13) has exponent field E + 127.
__device__ __forceinline__ uint32_t pbits64_raw(uint32_t f) { return f * 1400832u + 907255808u; }  // (f*171+110749)<<13

// PTX prmt in default mode: a selector nibble with bit 3 set replicates the sign bit of the selected byte
// (CUDA's __byte_perm masks the selector to 3 bits per nibble, so it cannot do this).
__device__ __forceinline__ uint32_t prmt_sign(uint32_t a) {
  uint32_t d;
  asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(d) : "r"(a));
  return d;
}

// Decode 4 E4M3 bytes (k..k+3 of one row).  sg: sign bits (0x80 per negative byte); la: la+9 per byte (LA_ZERO for
// zeros); x: E-pass byte 3*ea + 60 per byte (0 for zeros); v: values.  Pure arithmetic on the word: no memory access
// depends on the byte values, so every byte value (NaN codes included) is safe.
__device__ __forceinline__ void decode_word(uint32_t w, uint32_t& sg, uint32_t& la, uint32_t& x, float (&v)[4]) {
  sg = w & 0x80808080u;
  const uint32_t nzm = prmt_sign((w & 0x7f7f7f7fu) + 0x7f7f7f7fu);  // 0xFF per nonzero byte
  const uint32_t e4 = (w >> 3) & 0x0f0f0f0fu;
  const uint32_t u = ((e4 + 0x7f7f7f7fu) >> 7) & 0x01010101u;                     // [e != 0] per byte
  const uint32_t mb = w & 0x07070707u;
  const uint32_t sel = __byte_perm(mb | (mb >> 4), 0u, 0x4420u);                  // nibbles m0 m1 m2 m3
  const uint32_t tz = __byte_perm(0x00010003u, 0x00010002u, sel);                 // tz(8 | m) per byte
  const uint32_t e1m1 = e4 - u;                                                    // clamped exponent + 6
  la = ((e1m1 + tz) & nzm) | (0x3c3c3c3cu & ~nzm);
  x = (e1m1 * 3u + 0x2a2a2a2au) & nzm;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(MMAEMU_V9_PORTABLE_DECODE)
  // hardware E4M3x2 -> f16x2 conversion (sm_89+), then f16 -> f32 (both exact); the f16 -> f32 step is the inline PTX
  // of cuda_fp16.hpp __half22float2 (CUDA 12.8), copied so that the source needs no CUDA headers under NVRTC.
  uint32_t h01, h23;
  asm("{.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, lo;\n\tcvt.rn.f16x2.e4m3x2 %1, hi;}"
      : "=r"(h01), "=r"(h23) : "r"(w));
  float f0, f1, f2, f3;
  asm("{.reg .f16 low,high;\n  mov.b32 {low,high},%1;\n  cvt.f32.f16 %0, low;}\n" : "=f"(f0) : "r"(h01));
  asm("{.reg .f16 low,high;\n  mov.b32 {low,high},%1;\n  cvt.f32.f16 %0, high;}\n" : "=f"(f1) : "r"(h01));
  asm("{.reg .f16 low,high;\n  mov.b32 {low,high},%1;\n  cvt.f32.f16 %0, low;}\n" : "=f"(f2) : "r"(h23));
  asm("{.reg .f16 low,high;\n  mov.b32 {low,high},%1;\n  cvt.f32.f16 %0, high;}\n" : "=f"(f3) : "r"(h23));
  v[0] = f0; v[1] = f1; v[2] = f2; v[3] = f3;
#else
  // portable path (pre-sm_89): the per-byte value decode (exhaustively tested in p2 as decode().v)
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = decode_value((w >> (8 * q)) & 0xffu);
#endif
}

}  // namespace mmaemu_v9
