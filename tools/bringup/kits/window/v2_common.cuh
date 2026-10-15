// Shared device primitives of the p2-int-mixed kernels (included by v2.cu, v3.cu and prims.cu so that the exhaustive
// primitive tests exercise exactly the code the GEMM kernels run).
#pragma once
#include <stdint.h>
#include <cuda_fp16.h>

namespace p2 {

constexpr uint32_t LA_ZERO = 60;  // LA/LB byte code of a zero operand (valid for block exponents E <= 55)

__device__ __forceinline__ uint32_t bfma2(uint32_t a, uint32_t b, uint32_t c) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  __trap();  // bf16x2 FMA needs sm_80+: use the fp32 E-pass (v4 EMODE 1) on older GPUs
  return 0;
#endif
}
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

// E4M3 byte decode (NaN codes 0x7F/0xFF unsupported).
//   v:   value (fp32)
//   o32: fp32 bits of 2^(5*ea) or 0 for zero               (fp32 E-pass)
//   o16: bf16 bits of 2^(5*ea - 30) or 0 for zero          (bf16x2 E-pass)
//   la:  la + 9 (la = exponent of the lowest set bit of |x|) or LA_ZERO for zero;  sg: 0x80 if the sign bit is set
struct Dec { float v; uint32_t o32, o16, la, sg; };

__device__ __forceinline__ Dec decode(uint32_t byte) {
  Dec d;
  const uint32_t e = (byte >> 3) & 15u, m = byte & 7u;
  const bool zero = (byte & 0x7fu) == 0u;
  const uint32_t e1 = e ? e : 1u;  // clamped exponent + 7  (ea = e1 - 7)
  const float mag = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
  d.v = (byte & 0x80u) ? -mag : mag;
  d.o32 = zero ? 0u : ((5u * e1 - 35u + 127u) << 23);
  d.o16 = zero ? 0u : ((5u * e1 - 35u - 30u + 127u) << 7);
  const uint32_t tz = (0x1213u >> (2u * m)) & 3u;  // trailing zeros of (8 | m)
  d.la = zero ? LA_ZERO : (e1 - 1u + tz);          // la + 9 = (e1 - 7 - 3 + tz) + 9
  d.sg = (byte & 0x80u) ? 0x80u : 0u;
  return d;
}

// E-pass accumulator init: one extra term at the minimum product exponent (-12), so E_prod >= -12 always.
constexpr uint32_t EPASS_INIT_BF16X2 = 0x03800380u;  // bf16 2^-120 = 2^(5*(-12) - 60) in both lanes
constexpr float EPASS_INIT_F32 = 0x1p-60f;           // 2^(5*(-12))

// fp32 bits of 2^E_prod from the exponent field f of the base-32 sum (sum in [32^E, 18*32^E): f = 5E + bias + d,
// d in [0, 4]):  bf16: f = 5E + 67 + d;  fp32: f = 5E + 127 + d.   E + 12 = floor((f - bias + 60) / 5),
// floor(x / 5) = (x * 205) >> 10 for 0 <= x < 1020.   Pbits = (E + 127) << 23.
__device__ __forceinline__ uint32_t pbits_bf16(uint32_t f) { return ((f * 205u + 116325u) << 13) & 0x7f800000u; }
__device__ __forceinline__ uint32_t pbits_f32(uint32_t f) { return (((f - 60u) * 205u + 116325u) << 13) & 0x7f800000u; }

// Block start: E = max(E_prod, exponent(V)); returns S0 = 1.5*2^(E+10) + trunc(V to the grid 2^(E-13)) (exact).
__device__ __forceinline__ float block_init(float V, uint32_t pb) {
  const uint32_t e2 = max(pb, __float_as_uint(V) & 0x7f800000u);                         // bits of 2^E
  const float Mm = __uint_as_float(e2 + ((10u << 23) | 0x400000u));                        // 1.5 * 2^(E+10)
  const float mc = __uint_as_float((__float_as_uint(V) & 0x80000000u) | (e2 + (10u << 23)));  // +-2^(E+10)
  const float ct = __fsub_rn(__fadd_rz(V, mc), mc);                                        // trunc(V) on the grid
  return __fadd_rn(ct, Mm);
}

// Block end: S = 1.5*2^(E+10) + c_t + sum floor(p_k) (exact);  la/sa: A-side byte words of the 4 k-groups of this
// block, lb/sb: B side.  Adds the number of negative inexact products and returns V = S - 1.5*2^(E+10).
__device__ __forceinline__ float block_finish(float S, const uint32_t (&la)[4], const uint32_t (&sa)[4],
                                              const uint32_t (&lb)[4], const uint32_t (&sb)[4]) {
  const uint32_t s = __float_as_uint(S);
  const uint32_t tt = (260u - (s >> 23)) * 0x01010101u;  // byte 123 - E  (S exponent field = E + 137)
  uint32_t corr = 0;
#pragma unroll
  for (int q = 0; q < 4; ++q) corr = dp4a_u(lop3_corr(la[q] + lb[q] + tt, sa[q], sb[q]), 0x01010101u, corr);
  const uint32_t s2 = s + (corr >> 7);
  const float Mx = __uint_as_float((s2 & 0xff800000u) | 0x400000u);
  return __fsub_rn(__uint_as_float(s2), Mx);
}

// After the last block: truncate V to 14 significant bits (13 fraction bits relative to its exponent).
__device__ __forceinline__ float final_trunc(float V) {
  const float m2 = __uint_as_float((__float_as_uint(V) & 0xff800000u) + (10u << 23));
  return __fsub_rn(__fadd_rz(V, m2), m2);
}

}  // namespace p2

// ------------------------------------------------------------------------------------------------------------------
// v3 primitives: word-level (4 E4M3 bytes = 4 consecutive k) SIMD decode and a base-64 bf16x2 E-pass whose operands
// are built by byte placement (PRMT): a bf16 half with bits 8-15 = x (x < 128), bits 0-7 = 0 is 2^(2x - 127).
// With x = 3*ea + 60 (ea = clamped exponent) a product of two operands is 2^(6*(ea+eb) - 14); <= 17 terms (16 + init)
// keep the sum below 18 * 64^E, so the exponent field is 6E + 113 + d with d in [0, 4].
namespace p2 {

constexpr uint32_t EP64_INIT = 0x14801480u;  // bf16 2^-86 = 2^(6*(-12) - 14) in both lanes

// fp32 bits of 2^E_prod (mantissa bits = garbage, exponent exact) from the base-64 field f (0 <= f < 256):
// E + 12 = floor((f - 41) / 6) = ((f - 41) * 171) >> 10;  ((f*171 + 110749) << 13) has exponent field E + 127.
__device__ __forceinline__ uint32_t pbits64_raw(uint32_t f) { return f * 1400832u + 907255808u; }  // (f*171+110749)<<13

// Decode 4 E4M3 bytes (k..k+3 of one row).  sg: sign bits (0x80 per negative byte); la: la+9 per byte (LA_ZERO for
// zeros); x: E-pass byte 3*ea + 60 per byte (0 for zeros); v: values.
// PTX prmt in default mode: a selector nibble with bit 3 set replicates the sign bit of the selected byte
// (CUDA's __byte_perm masks the selector to 3 bits per nibble, so it cannot do this).
__device__ __forceinline__ uint32_t prmt_sign(uint32_t a) {
  uint32_t d;
  asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(d) : "r"(a));
  return d;
}

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
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890) && !defined(P2_PORTABLE_DECODE)
  // hardware E4M3x2 -> f16x2 conversion (sm_89+), then f16 -> f32 (both exact)
  uint32_t h01, h23;
  asm("{.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, lo;\n\tcvt.rn.f16x2.e4m3x2 %1, hi;}"
      : "=r"(h01), "=r"(h23) : "r"(w));
  float2 f01 = __half22float2(*reinterpret_cast<__half2*>(&h01));
  float2 f23 = __half22float2(*reinterpret_cast<__half2*>(&h23));
  v[0] = f01.x; v[1] = f01.y; v[2] = f23.x; v[3] = f23.y;
#else
  // portable path (pre-sm_89): the per-byte value decode of decode() (exhaustively tested)
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = decode((w >> (8 * q)) & 0xffu).v;
#endif
}

// A-side E-pass operand for byte q of x: bf16x2 {2^(2x_q-127), 2^(2x_q-127)} = bytes [0, x_q, 0, x_q]
template <int Q> __device__ __forceinline__ uint32_t epa(uint32_t x) { return __byte_perm(x, 0u, 0x0404u | (Q << 4) | (Q << 12)); }
// B-side: the 16-bit half [0, x_q] (stored as uint16 next to the neighbouring column's half)
template <int Q> __device__ __forceinline__ uint32_t epb(uint32_t x) { return __byte_perm(x, 0u, 0x4404u | (Q << 4)); }

// Block start (v3): E = max(E_prod, exponent(V)) via a float max (|V| and pb_raw are positive floats whose exponents
// are exponent(V) and E_prod); S0 = 1.5*2^(E+10) + trunc(V to the grid 2^(E-13)), magic +-1.5*2^(E+10).
__device__ __forceinline__ float block_init3(float V, uint32_t pb_raw) {
  const uint32_t e2 = __float_as_uint(fmaxf(fabsf(V), __uint_as_float(pb_raw))) & 0x7f800000u;
  const uint32_t mm = e2 + 0x05400000u;                                          // 1.5 * 2^(E+10)
  const float mc = __uint_as_float((__float_as_uint(V) & 0x80000000u) | mm);     // sign(V) * 1.5 * 2^(E+10)
  const float ct = __fsub_rn(__fadd_rz(V, mc), mc);
  return __fadd_rn(ct, __uint_as_float(mm));
}

}  // namespace p2
