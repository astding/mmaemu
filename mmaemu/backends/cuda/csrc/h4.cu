// mmaemu kernel h4: the FP8 E4M3 x E4M3 accumulation chain of cuBLAS on the H100 PCIe (preset "hopper") on CUDA
// cores, IEEE-defined instructions only (fp32 FFMA.RZ chain, fma.rn.bf16x2 exponent pass, integer byte SIMD
// corrections): bit-exact on every sm_80+ GPU. Kernel ABI v1 (spec.py). The arithmetic is kernel v4's (v4.cu) with
// 32-term blocks instead of 16; the model is h1's (h1.cu; MATLAB Tensor Core toolbox, BSD 2-Clause, Copyright (c)
// 2025 Faizan A. Khattak and Mantas Mikaitis; see NOTICE).
//
// Algorithm per 32-term block, as v4's per 16-term block: the block exponent E = max(E_prod, exponent(acc)) comes
// from a base-64 bf16x2 power sum; the accumulator is truncated to the grid 2^(E-13) and offset by 1.5 * 2^(E+10);
// products are added with FFMA.RZ (floor of every product on that grid); a byte-SIMD count of the negative inexact
// products turns floor into truncation toward zero; the final value is truncated to 14 significant bits.
//
// Why v4's arithmetic holds for 32 terms. Products satisfy |p| < 2^(E+2) (each significand is below 2) and the
// accumulator |acc| < 2^(E+1).
// - Exponent pass: the sum of at most 33 powers of two 2^(6 e) (32 products and the 2^-86 floor) lies in
//   [64^E, 64^(E+1)) even after bf16 round-to-nearest (at most 33 * (1 + 2^-8)^33 < 38 times the largest), so its
//   exponent field is 6E + 113 + d with d in [0, 5], and pbits64_raw recovers E for every d <= 5.
// - Chain: 1.5 * 2^(E+10) plus the truncated accumulator plus 32 floored products differs from 1.5 * 2^(E+10) by
//   less than 130 * 2^E < 2^(E+9), so every partial sum stays in [2^(E+10), 2^(E+11)) with ulp 2^(E-13): each
//   FFMA.RZ is an exact floor on the grid.
// - Correction: decided per product from the byte encodings (8 k-groups of 4 instead of v4's 4), at most 32 ulps
//   added to the bit pattern of the partial sum, which stays in its binade.
//
// Layout: tile 128 x 128, 256 threads, 8 x 8 outputs per thread, one 32-term block per 32-wide k-tile, one stage
// (the next k-tile is prefetched into registers). K % 32 == 16 leaves a half block: its other 16 terms are loaded
// as zero codes, which the model excludes, so the result equals the zero-padded chain.
//
// Preconditions (checked by the Python wrapper, never here): 1 <= M, N; 16 <= K < 2^24 (keeps the block exponent
// <= 55 in the byte encodings); K % 16 == 0; all codes finite (NaN codes give unspecified values, never a fault);
// A, B 4-byte aligned; grid (ceil(N/128), ceil(M/128), 1), block (256, 1, 1), dynamic shared memory
// sizeof(Smem) = 73728 bytes (above the 48 KiB default: the launcher opts in per function).
#include "common.cuh"

namespace {
using namespace mmaemu;

// LA/LB byte code of a zero operand: 60 = 0x3c (valid for block exponents E <= 55); see decode_word.

__device__ __forceinline__ uint32_t bfma2(uint32_t a, uint32_t b, uint32_t c) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  __trap();  // bf16x2 FMA needs sm_80+ (h4 is only built for sm_80+)
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
// Portable per-byte E4M3 value decode (pre-sm_89 path of decode_word; NaN codes unsupported).
__device__ __forceinline__ float decode_value(uint32_t byte) {
  const uint32_t e = (byte >> 3) & 15u, m = byte & 7u;
  const float mag = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
  return (byte & 0x80u) ? -mag : mag;
}
#endif

constexpr int NB = 32;                       // terms per block (preset hopper: nfma = 32)
constexpr uint32_t EP64_INIT = 0x14801480u;  // bf16 2^-86 = 2^(6*(-12) - 14) in both lanes

// fp32 bits of 2^E_prod (mantissa bits = garbage, exponent exact) from the base-64 field f (0 <= f < 256):
// E + 12 = floor((f - 41) / 6) = ((f - 41) * 171) >> 10;  ((f*171 + 110749) << 13) has exponent field E + 127.
__device__ __forceinline__ uint32_t pbits64_raw(uint32_t f) { return f * 1400832u + 907255808u; }

// PTX prmt in default mode: a selector nibble with bit 3 set replicates the sign bit of the selected byte.
__device__ __forceinline__ uint32_t prmt_sign(uint32_t a) {
  uint32_t d;
  asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(d) : "r"(a));
  return d;
}

// Decode 4 E4M3 bytes (k..k+3 of one row).  sg: sign bits (0x80 per negative byte); la: la+9 per byte (LA_ZERO for
// zeros); x: E-pass byte 3*ea + 60 per byte (0 for zeros); v: values.
__device__ __forceinline__ void decode_word(uint32_t w, uint32_t& sg, uint32_t& la, uint32_t& x, float (&v)[4]) {
  sg = w & 0x80808080u;
  const uint32_t nzm = prmt_sign((w & 0x7f7f7f7fu) + 0x7f7f7f7fu);  // 0xFF per nonzero byte
  const uint32_t e4 = (w >> 3) & 0x0f0f0f0fu;
  const uint32_t u = ((e4 + 0x7f7f7f7fu) >> 7) & 0x01010101u;  // [e != 0] per byte
  const uint32_t mb = w & 0x07070707u;
  const uint32_t sel = __byte_perm(mb | (mb >> 4), 0u, 0x4420u);  // nibbles m0 m1 m2 m3
  const uint32_t tz = __byte_perm(0x00010003u, 0x00010002u, sel);  // tz(8 | m) per byte
  const uint32_t e1m1 = e4 - u;                                    // clamped exponent + 6
  la = ((e1m1 + tz) & nzm) | (0x3c3c3c3cu & ~nzm);
  x = (e1m1 * 3u + 0x2a2a2a2au) & nzm;
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  // hardware E4M3x2 -> f16x2 conversion (sm_89+), then f16 -> f32 (both exact)
  uint32_t h01, h23;
  asm("{.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, lo;\n\tcvt.rn.f16x2.e4m3x2 %1, hi;}"
      : "=r"(h01), "=r"(h23) : "r"(w));
  v[0] = f16_bits_to_f32(h01 & 0xffffu); v[1] = f16_bits_to_f32(h01 >> 16);
  v[2] = f16_bits_to_f32(h23 & 0xffffu); v[3] = f16_bits_to_f32(h23 >> 16);
#else
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = decode_value((w >> (8 * q)) & 0xffu);
#endif
}

// A-side E-pass operand for byte q of x: bf16x2 {2^(2x_q-127), 2^(2x_q-127)} = bytes [0, x_q, 0, x_q]
template <int Q> __device__ __forceinline__ uint32_t epa(uint32_t x) { return __byte_perm(x, 0u, 0x0404u | (Q << 4) | (Q << 12)); }
// B-side: the 16-bit half [0, x_q] (stored as uint16 next to the neighbouring column's half)
template <int Q> __device__ __forceinline__ uint32_t epb(uint32_t x) { return __byte_perm(x, 0u, 0x4404u | (Q << 4)); }

// Block start: E = max(E_prod, exponent(V)) via a float max (|V| and pb_raw are positive floats whose exponents
// are exponent(V) and E_prod); S0 = 1.5*2^(E+10) + trunc(V to the grid 2^(E-13)), magic +-1.5*2^(E+10).
__device__ __forceinline__ float block_init3(float V, uint32_t pb_raw) {
  const uint32_t e2 = __float_as_uint(fmaxf(fabsf(V), __uint_as_float(pb_raw))) & 0x7f800000u;
  const uint32_t mm = e2 + 0x05400000u;                                        // 1.5 * 2^(E+10)
  const float mc = __uint_as_float((__float_as_uint(V) & 0x80000000u) | mm);   // sign(V) * 1.5 * 2^(E+10)
  const float ct = __fsub_rn(__fadd_rz(V, mc), mc);
  return __fadd_rn(ct, __uint_as_float(mm));
}

// Bytes 123 - E of S = 1.5*2^(E+10) + ... (S exponent field = E + 137): with la + lb, bit 7 of a byte is set exactly
// when that product is exact on the grid 2^(E-13) (v4.cu's block_finish).
__device__ __forceinline__ uint32_t block_tt(float S) { return (260u - (__float_as_uint(S) >> 23)) * 0x01010101u; }

// Adds to S = 1.5*2^(E+10) + c_t + sum floor(p_k) the number of negative inexact products among the 4 of one
// k-group (la/sa: A-side byte words, lb/sb: B side, tt = block_tt(S)): the lop3 result has only bit 7 set per
// counted byte, so shifted right by 7 it holds 0 or 1 per byte, and dp4a adds those to S's bit pattern. S stays in
// its binade, so tt is the same for every group.
__device__ __forceinline__ float add_negative_inexact(float S, uint32_t la, uint32_t sa, uint32_t lb, uint32_t sb,
                                                      uint32_t tt) {
  return __uint_as_float(dp4a_u(lop3_corr(la + lb + tt, sa, sb) >> 7, 0x01010101u, __float_as_uint(S)));
}

// Block end: V = S - 1.5*2^(E+10) (exact).
__device__ __forceinline__ float block_end(float S) {
  const float Mx = __uint_as_float((__float_as_uint(S) & 0xff800000u) | 0x400000u);
  return __fsub_rn(S, Mx);
}

// After the last block: truncate V to 14 significant bits (13 fraction bits relative to its exponent).
__device__ __forceinline__ float final_trunc(float V) {
  const float m2 = __uint_as_float((__float_as_uint(V) & 0xff800000u) + (10u << 23));
  return __fsub_rn(__fadd_rz(V, m2), m2);
}

template <int BM, int BN, int KT>
struct Smem {
  static constexpr int KG = KT / 4;
  float Av[KT][BM];
  uint32_t Ao[KT][BM];
  uint32_t Al[KG][BM];
  uint32_t As[KG][BM];
  float Bv[KT][BN];
  uint16_t Bo[KT][BN];
  uint32_t Bl[KG][BN];
  uint32_t Bs[KG][BN];
};

template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void h4_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                        float* __restrict__ C, int M, int N, int K) {
  constexpr int KT = NB;  // one block per k-tile
  constexpr int NT = (BM / TM) * (BN / TN);
  constexpr int KG = KT / 4;
  constexpr int WA = BM * KG / NT, WB = BN * KG / NT;
  static_assert(BM * KG % NT == 0 && BN * KG % NT == 0, "loader mapping");
  using SM = Smem<BM, BN, KT>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  SM& s = *reinterpret_cast<SM*>(smem_raw);

  const int tid = threadIdx.x;
  const int tn = tid % (BN / TN), tm = tid / (BN / TN);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  uint32_t ra[WA], rb[WB];
  auto gload = [&](int k0) {
#pragma unroll
    for (int u = 0; u < WA; ++u) {
      const int idx = tid + u * NT, r = idx % BM, w = idx / BM, gm = m0 + r, gk = k0 + 4 * w;
      ra[u] = (gm < M && gk < K) ? __ldg(reinterpret_cast<const uint32_t*>(A + (size_t)gm * K + gk)) : 0u;
    }
#pragma unroll
    for (int u = 0; u < WB; ++u) {
      const int idx = tid + u * NT, r = idx % BN, w = idx / BN, gn = n0 + r, gk = k0 + 4 * w;
      rb[u] = (gn < N && gk < K) ? __ldg(reinterpret_cast<const uint32_t*>(B + (size_t)gn * K + gk)) : 0u;
    }
  };
  auto sstore = [&]() {
#pragma unroll
    for (int u = 0; u < WA; ++u) {
      const int idx = tid + u * NT, r = idx % BM, w = idx / BM;
      uint32_t sg, la, x; float v[4];
      decode_word(ra[u], sg, la, x, v);
      s.Av[4 * w + 0][r] = v[0]; s.Av[4 * w + 1][r] = v[1]; s.Av[4 * w + 2][r] = v[2]; s.Av[4 * w + 3][r] = v[3];
      s.Ao[4 * w + 0][r] = epa<0>(x); s.Ao[4 * w + 1][r] = epa<1>(x);
      s.Ao[4 * w + 2][r] = epa<2>(x); s.Ao[4 * w + 3][r] = epa<3>(x);
      s.Al[w][r] = la; s.As[w][r] = sg;
    }
#pragma unroll
    for (int u = 0; u < WB; ++u) {
      const int idx = tid + u * NT, r = idx % BN, w = idx / BN;
      uint32_t sg, la, x; float v[4];
      decode_word(rb[u], sg, la, x, v);
      s.Bv[4 * w + 0][r] = v[0]; s.Bv[4 * w + 1][r] = v[1]; s.Bv[4 * w + 2][r] = v[2]; s.Bv[4 * w + 3][r] = v[3];
      s.Bo[4 * w + 0][r] = (uint16_t)epb<0>(x); s.Bo[4 * w + 1][r] = (uint16_t)epb<1>(x);
      s.Bo[4 * w + 2][r] = (uint16_t)epb<2>(x); s.Bo[4 * w + 3][r] = (uint16_t)epb<3>(x);
      s.Bl[w][r] = la; s.Bs[w][r] = sg;
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  auto compute = [&]() {
    uint32_t Pb[TM][TN];
    {
      uint32_t ea[TM][TN / 2];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int jp = 0; jp < TN / 2; ++jp) ea[i][jp] = EP64_INIT;
#pragma unroll
      for (int c = 0; c < NB; ++c) {
        uint32_t oa[TM], ob[TN / 2];
#pragma unroll
        for (int i = 0; i < TM; ++i) oa[i] = s.Ao[c][tm * TM + i];
#pragma unroll
        for (int jp = 0; jp < TN / 2; ++jp) ob[jp] = reinterpret_cast<const uint32_t*>(&s.Bo[c][0])[tn * (TN / 2) + jp];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int jp = 0; jp < TN / 2; ++jp) ea[i][jp] = bfma2(oa[i], ob[jp], ea[i][jp]);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int jp = 0; jp < TN / 2; ++jp) {
          const uint32_t w = ea[i][jp];
          Pb[i][2 * jp] = pbits64_raw((w >> 7) & 0xffu);
          Pb[i][2 * jp + 1] = pbits64_raw(w >> 23);
        }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = block_init3(acc[i][j], Pb[i][j]);
#pragma unroll
    for (int c = 0; c < NB; ++c) {
      float va[TM], vb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) va[i] = s.Av[c][tm * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) vb[j] = s.Bv[c][tn * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = __fmaf_rz(va[i], vb[j], acc[i][j]);
    }
    // negative inexact products, one k-group at a time: holding the byte words of all 8 groups live, as v4 does
    // for its 4, needs too many registers
    uint32_t tt[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) tt[i][j] = block_tt(acc[i][j]);
#pragma unroll
    for (int q = 0; q < KG; ++q) {
      uint32_t la[TM], sa[TM], lb[TN], sb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) { la[i] = s.Al[q][tm * TM + i]; sa[i] = s.As[q][tm * TM + i]; }
#pragma unroll
      for (int j = 0; j < TN; ++j) { lb[j] = s.Bl[q][tn * TN + j]; sb[j] = s.Bs[q][tn * TN + j]; }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = add_negative_inexact(acc[i][j], la[i], sa[i], lb[j], sb[j], tt[i][j]);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = block_end(acc[i][j]);
  };

  gload(0);
#pragma unroll 1
  for (int k0 = 0; k0 < K; k0 += KT) {
    sstore();
    __syncthreads();
    if (k0 + KT < K) gload(k0 + KT);
    compute();
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gm = m0 + tm * TM + i, gn = n0 + tn * TN + j;
      if (gm < M && gn < N) C[(size_t)gm * N + gn] = final_trunc(acc[i][j]);
    }
}

}  // namespace

// Kernel ABI v1 (spec.py). Tile 128 x 128, 256 threads, 1 stage, dynamic shared memory 73728 bytes.
extern "C" __global__ void __launch_bounds__(256, 1)
mmaemu_h4_e4m3e4m3(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
                   int M, int N, int K) {
  h4_body<128, 128, 8, 8>(A, B, C, M, N, K);
}
