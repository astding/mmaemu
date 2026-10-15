// mmaemu kernel v1: Ada (sm_89) FP8 E4M3 x E4M3 tensor-core accumulation chain, emulated on the FP32 pipe.
// Portable (IEEE-defined instructions only): bit-exact on every GPU it runs on. Kernel ABI v1 (spec.py).
//
// Model (preset "ada"): per 16-term block, the products and the running accumulator are aligned to the largest
// exponent E among them (product exponent = ea + eb, input exponents clamped at -6; zero products excluded),
// truncated toward zero to 13 fraction bits below 2^E, summed exactly, and the sum is truncated to 13 fraction bits
// relative to its own exponent. Blocks are chained along K starting from +0. An exactly-zero block result is +0.
// Translated and adapted from the MATLAB Tensor Core toolbox (BSD 2-Clause, Copyright (c) 2025 Faizan A. Khattak and
// Mantas Mikaitis; https://github.com/north-numerical-computing/MATLAB-tensor-core); see NOTICE.
//
// Arithmetic: products of E4M3 values are exact in fp32. Truncating x toward zero to the grid 2^(E-13) is an exact
// round-toward-zero add of M = copysign(2^(E+10), x) (ulp 2^(E-13)) followed by an exact subtraction of M. Block sums
// are multiples of 2^(E-13) below 2^(E+7): exact in fp32. Block maximum exponents use packed byte SIMD (4 products
// per instruction).
//
// Origin: tcemu csrc/ada_fp8_gemm.cu, kernel v1, configuration cfg4 (BM 64, BN 64, TM 4, TN 4, KT 16, 256 threads),
// with the audit fixes: NaN codes decode to NaN (verify-c1-claims-3, B2); compile-time __constant__ tables
// (verify-a2-kernel-review-3/f1.patch, B3). Decode variants (verify-p0-profile-0 DEC1 shared-memory LUT,
// verify-p0-profile-1 exp1b carry-free exponent bytes) are selected at compile time below; the shipped defaults were
// chosen by the rule "bitwise identical and >= 1.2x faster on sm_89" (docs/backends.md).
//
// Preconditions (checked by the Python wrapper, never here): 1 <= M, N; 16 <= K; K % 16 == 0; A, B 4-byte aligned;
// grid (ceil(N/64), ceil(M/64), 1), block (256, 1, 1), no dynamic shared memory. Any byte value is safe.
#include "common.cuh"

// Shipped: DEC1 (bitwise equal, 1.235x faster than DEC0 on sm_89); not shipped: exp1b (only 1.138x on top of DEC1).
// Measurements and the decision rule: docs/backends.md, "v1 variant decision".
#ifndef MMAEMU_V1_DEC
#define MMAEMU_V1_DEC 1  // 0: values from the __constant__ table; 1: per-block shared-memory copy of the table (DEC1)
#endif
#ifndef MMAEMU_V1_EXP
#define MMAEMU_V1_EXP 0  // 0: signed exponent bytes, __vadd4/__vmaxs4; 1: carry-free biased bytes (exp1b)
#endif

namespace {
using namespace mmaemu;

__constant__ uint32_t c_valbits[256] = {MMAEMU_T256(e4m3_val_bits)};  // fp32 bits of the E4M3 value (NaN canonical)
#if MMAEMU_V1_EXP == 0
__constant__ int c_exp8[256] = {MMAEMU_T256(e4m3_exp8)};               // alignment exponent (zero / NaN: ZERO_EXP8)
#else
// exp1b: exponent bytes stored biased by EXP_BIAS: ee + 39 in [33, 47], zero and NaN codes -> 0, so the per-byte sums
// ea + eb are <= 94 and the packed byte maximum needs no carry handling.
constexpr int EXP_BIAS = 39;
__device__ __forceinline__ int exp_byte_biased(int byte) {
  const int e = (byte >> 3) & 15, m = byte & 7;
  return e == 0 ? (m == 0 ? 0 : 33) : ((e == 15 && m == 7) ? 0 : e + 32);
}
#endif

__device__ __forceinline__ float trunc_to(float x, int mbits) {  // mbits = bit pattern of 2^(E+10)
  const float m = __int_as_float(mbits | (__float_as_int(x) & 0x80000000));
  return __fsub_rn(__fadd_rz(x, m), m);
}

__device__ __forceinline__ int fexp(float x) { return ((__float_as_int(x) >> 23) & 0xff) - 127; }  // x == 0 -> -127

template <int BM, int BN, int TM, int TN, int KT>
__device__ __forceinline__ void v1_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                        float* __restrict__ C, int M, int N, int K) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float sAv[KT][BM];
  __shared__ float sBv[KT][BN];
  __shared__ int sAe[KT / 4][BM];  // 4 packed int8 exponents (k .. k+3)
  __shared__ int sBe[KT / 4][BN];
#if MMAEMU_V1_DEC == 1
  __shared__ uint32_t s_val[256];
#if MMAEMU_V1_EXP == 0
  __shared__ int s_exp8[256];
#endif
#endif
  const int tid = threadIdx.x;
  const int tn = tid % (BN / TN), tm = tid / (BN / TN);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
#if MMAEMU_V1_DEC == 1
  for (int i = tid; i < 256; i += NT) {
    s_val[i] = c_valbits[i];
#if MMAEMU_V1_EXP == 0
    s_exp8[i] = c_exp8[i];
#endif
  }
  __syncthreads();
#define MMAEMU_VAL(byte) __uint_as_float(s_val[byte])
#define MMAEMU_EXP8(byte) s_exp8[byte]
#else
#define MMAEMU_VAL(byte) __uint_as_float(c_valbits[byte])
#define MMAEMU_EXP8(byte) c_exp8[byte]
#endif
#if MMAEMU_V1_EXP == 0
#define MMAEMU_EXPBYTE(byte) (MMAEMU_EXP8(byte) & 0xff)
#else
#define MMAEMU_EXPBYTE(byte) exp_byte_biased(byte)
#endif

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KT) {
    for (int idx = tid; idx < BM * (KT / 4); idx += NT) {
      const int r = idx / (KT / 4), w = idx % (KT / 4), gm = m0 + r, gk = k0 + 4 * w;
      const uint32_t word = (gm < M && gk < K) ? *reinterpret_cast<const uint32_t*>(A + (size_t)gm * K + gk) : 0u;
      int packed = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int byte = (word >> (8 * q)) & 0xff;
        sAv[4 * w + q][r] = MMAEMU_VAL(byte);
        packed |= MMAEMU_EXPBYTE(byte) << (8 * q);
      }
      sAe[w][r] = packed;
    }
    for (int idx = tid; idx < BN * (KT / 4); idx += NT) {
      const int r = idx / (KT / 4), w = idx % (KT / 4), gn = n0 + r, gk = k0 + 4 * w;
      const uint32_t word = (gn < N && gk < K) ? *reinterpret_cast<const uint32_t*>(B + (size_t)gn * K + gk) : 0u;
      int packed = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int byte = (word >> (8 * q)) & 0xff;
        sBv[4 * w + q][r] = MMAEMU_VAL(byte);
        packed |= MMAEMU_EXPBYTE(byte) << (8 * q);
      }
      sBe[w][r] = packed;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < KT; g += 16) {
      int mb[TM][TN];  // bit pattern of 2^(E+10) per output
      {
        int mx[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) mx[i][j] = MMAEMU_V1_EXP == 0 ? (int)0x80808080 : 0;  // below every product
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          int ea[TM], eb[TN];
#pragma unroll
          for (int i = 0; i < TM; ++i) ea[i] = sAe[g / 4 + w][tm * TM + i];
#pragma unroll
          for (int j = 0; j < TN; ++j) eb[j] = sBe[g / 4 + w][tn * TN + j];
#pragma unroll
          for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) {
#if MMAEMU_V1_EXP == 0
              mx[i][j] = __vmaxs4(mx[i][j], __vadd4(ea[i], eb[j]));
#else
              const uint32_t s = (uint32_t)ea[i] + (uint32_t)eb[j];       // per byte <= 94: no carries
              const uint32_t t = (uint32_t)mx[i][j] + 0x80808080u - s;    // per byte in [34, 222]: bit7 = (mx >= s)
              const uint32_t keep = prmt(t, 0, 0xba98);                    // 0xff in bytes where mx >= s
              mx[i][j] = (int)(((uint32_t)mx[i][j] & keep) | (s & ~keep));  // byte-wise max
#endif
            }
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const int x = mx[i][j];
#if MMAEMU_V1_EXP == 0
            int e = max(max((int)(signed char)(x & 0xff), (int)(signed char)((x >> 8) & 0xff)),
                        max((int)(signed char)((x >> 16) & 0xff), (int)(signed char)(x >> 24)));
#else
            int e = max(max((int)prmt(x, 0, 0x4440), (int)prmt(x, 0, 0x4441)),
                        max((int)prmt(x, 0, 0x4442), (int)prmt(x, 0, 0x4443))) - 2 * EXP_BIAS;
#endif
            e = max(max(e, fexp(acc[i][j])), -100);
            mb[i][j] = (e + 10 + 127) << 23;
          }
      }
      float S[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) S[i][j] = trunc_to(acc[i][j], mb[i][j]);
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        float va[TM], vb[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) va[i] = sAv[g + c][tm * TM + i];
#pragma unroll
        for (int j = 0; j < TN; ++j) vb[j] = sBv[g + c][tn * TN + j];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) S[i][j] = __fadd_rn(S[i][j], trunc_to(__fmul_rn(va[i], vb[j]), mb[i][j]));
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = trunc_to(S[i][j], (fexp(S[i][j]) + 10 + 127) << 23);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gm = m0 + tm * TM + i, gn = n0 + tn * TN + j;
      if (gm < M && gn < N) C[(size_t)gm * N + gn] = acc[i][j];
    }
#undef MMAEMU_VAL
#undef MMAEMU_EXP8
#undef MMAEMU_EXPBYTE
}

}  // namespace

// Kernel ABI v1 (spec.py). Tile 64 x 64, 256 threads.
extern "C" __global__ void __launch_bounds__(256)
mmaemu_v1_e4m3e4m3(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
                   int M, int N, int K) {
  v1_body<64, 64, 4, 4, 16>(A, B, C, M, N, K);
}
