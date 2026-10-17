// mmaemu kernel h1: the FP8 E4M3 x E4M3 accumulation chain of cuBLAS on the H100 PCIe (preset "hopper"), emulated
// on the FP32 pipe. Portable (IEEE-defined instructions only): bit-exact on every GPU it runs on. Kernel ABI v1
// (spec.py). The arithmetic is kernel v1's (v1.cu) with 32-term blocks instead of 16.
//
// Model (preset "hopper", mmaemu.model.chain): per 32-term block, the products and the running accumulator are
// aligned to the largest exponent E among them (product exponent = ea + eb, input exponents clamped at -6; zero
// products excluded), truncated toward zero to 13 fraction bits below 2^E, summed exactly, and the sum is truncated
// to 13 fraction bits relative to its own exponent. Blocks are chained along K starting from +0. An exactly-zero
// block result is +0. Translated and adapted from the MATLAB Tensor Core toolbox (BSD 2-Clause, Copyright (c) 2025
// Faizan A. Khattak and Mantas Mikaitis; https://github.com/north-numerical-computing/MATLAB-tensor-core); see NOTICE.
//
// Arithmetic: products of E4M3 values are exact in fp32. Truncating x toward zero to the grid 2^(E-13) is an exact
// round-toward-zero add of M = copysign(2^(E+10), x) (ulp 2^(E-13)) followed by an exact subtraction of M. A block
// sum has 33 terms (32 products and the accumulator), each below 2^(E+2) in magnitude (a product of two significands
// is below 4) and a multiple of 2^(E-13), so every partial sum is a multiple of 2^(E-13) below 2^(E+8): exact in
// fp32. Block maximum exponents use packed byte SIMD (4 products per instruction).
//
// Layout: tile 64 x 64, 256 threads, 4 x 4 outputs per thread, one 32-term block per k-tile; the decode tables are
// copied to shared memory per thread block (v1's DEC1 variant). K % 32 == 16 leaves a half block: its other 16
// terms are loaded as zero codes, which the model excludes, so the result equals the zero-padded chain.
//
// Preconditions (checked by the Python wrapper, never here): 1 <= M, N; 16 <= K; K % 16 == 0; A, B 4-byte aligned;
// grid (ceil(N/64), ceil(M/64), 1), block (256, 1, 1), no dynamic shared memory. Any byte value is safe.
#include "common.cuh"

namespace {
using namespace mmaemu;

__constant__ uint32_t c_valbits[256] = {MMAEMU_T256(e4m3_val_bits)};  // fp32 bits of the E4M3 value (NaN canonical)
__constant__ int c_exp8[256] = {MMAEMU_T256(e4m3_exp8)};               // alignment exponent (zero / NaN: ZERO_EXP8)

constexpr int NB = 32;  // terms per block (preset hopper: nfma = 32)

__device__ __forceinline__ float trunc_to(float x, int mbits) {  // mbits = bit pattern of 2^(E+10)
  const float m = __int_as_float(mbits | (__float_as_int(x) & 0x80000000));
  return __fsub_rn(__fadd_rz(x, m), m);
}

__device__ __forceinline__ int fexp(float x) { return ((__float_as_int(x) >> 23) & 0xff) - 127; }  // x == 0 -> -127

template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void h1_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                        float* __restrict__ C, int M, int N, int K) {
  constexpr int NT = (BM / TM) * (BN / TN);
  constexpr int KT = NB;           // one block per k-tile
  constexpr int KW = KT / 4;       // packed exponent words per block
  __shared__ float sAv[KT][BM];
  __shared__ float sBv[KT][BN];
  __shared__ int sAe[KW][BM];      // 4 packed int8 exponents (k .. k+3)
  __shared__ int sBe[KW][BN];
  __shared__ uint32_t s_val[256];
  __shared__ int s_exp8[256];
  const int tid = threadIdx.x;
  const int tn = tid % (BN / TN), tm = tid / (BN / TN);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  for (int i = tid; i < 256; i += NT) {
    s_val[i] = c_valbits[i];
    s_exp8[i] = c_exp8[i];
  }
  __syncthreads();

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KT) {
    for (int idx = tid; idx < BM * KW; idx += NT) {
      const int r = idx / KW, w = idx % KW, gm = m0 + r, gk = k0 + 4 * w;
      const uint32_t word = (gm < M && gk < K) ? *reinterpret_cast<const uint32_t*>(A + (size_t)gm * K + gk) : 0u;
      int packed = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int byte = (word >> (8 * q)) & 0xff;
        sAv[4 * w + q][r] = __uint_as_float(s_val[byte]);
        packed |= (s_exp8[byte] & 0xff) << (8 * q);
      }
      sAe[w][r] = packed;
    }
    for (int idx = tid; idx < BN * KW; idx += NT) {
      const int r = idx / KW, w = idx % KW, gn = n0 + r, gk = k0 + 4 * w;
      const uint32_t word = (gn < N && gk < K) ? *reinterpret_cast<const uint32_t*>(B + (size_t)gn * K + gk) : 0u;
      int packed = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int byte = (word >> (8 * q)) & 0xff;
        sBv[4 * w + q][r] = __uint_as_float(s_val[byte]);
        packed |= (s_exp8[byte] & 0xff) << (8 * q);
      }
      sBe[w][r] = packed;
    }
    __syncthreads();
    int mb[TM][TN];  // bit pattern of 2^(E+10) per output
    {
      int mx[TM][TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) mx[i][j] = (int)0x80808080;  // below every product
#pragma unroll
      for (int w = 0; w < KW; ++w) {
        int ea[TM], eb[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) ea[i] = sAe[w][tm * TM + i];
#pragma unroll
        for (int j = 0; j < TN; ++j) eb[j] = sBe[w][tn * TN + j];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) mx[i][j] = __vmaxs4(mx[i][j], __vadd4(ea[i], eb[j]));
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int x = mx[i][j];
          int e = max(max((int)(signed char)(x & 0xff), (int)(signed char)((x >> 8) & 0xff)),
                      max((int)(signed char)((x >> 16) & 0xff), (int)(signed char)(x >> 24)));
          e = max(max(e, fexp(acc[i][j])), -100);
          mb[i][j] = (e + 10 + 127) << 23;
        }
    }
    float S[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) S[i][j] = trunc_to(acc[i][j], mb[i][j]);
    // two unrolled halves: a full 32-way unroll spills 16 bytes on sm_80 (nvcc 12.8); 16 needs 63-96 registers and
    // no spills
#pragma unroll 16
    for (int c = 0; c < NB; ++c) {
      float va[TM], vb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) va[i] = sAv[c][tm * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) vb[j] = sBv[c][tn * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) S[i][j] = __fadd_rn(S[i][j], trunc_to(__fmul_rn(va[i], vb[j]), mb[i][j]));
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = trunc_to(S[i][j], (fexp(S[i][j]) + 10 + 127) << 23);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gm = m0 + tm * TM + i, gn = n0 + tn * TN + j;
      if (gm < M && gn < N) C[(size_t)gm * N + gn] = acc[i][j];
    }
}

}  // namespace

// Kernel ABI v1 (spec.py). Tile 64 x 64, 256 threads.
extern "C" __global__ void __launch_bounds__(256)
mmaemu_h1_e4m3e4m3(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
                   int M, int N, int K) {
  h1_body<64, 64, 4, 4>(A, B, C, M, N, K);
}
