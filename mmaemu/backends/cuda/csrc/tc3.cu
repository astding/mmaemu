// mmaemu tc3: Ada FP8 (E4M3 x E4M3) fast-accumulation chain on the fp16/bf16 tensor cores ("design A", dummy
// accumulator), kernel ABI v1 (mmaemu/backends/cuda/spec.py).  HARDWARE-DEPENDENT NUMERICS: exact only on GPUs whose
// fp16 HMMA (mma.sync.m16n8k8 f16 -> f32) has the Ampere/Ada numerics (8-term blocks, F = 24, c in the alignment) and
// whose bf16 HMMA has the window property.  It is shipped as SASS for sm_80 / sm_86 / sm_89 only and is used only after
// the per-device "design_a" gate (mmaemu/backends/cuda/fast/gates.py) has passed; validated on sm_89 (RTX 4060).
//
// Provenance: the tcemu audit's p3-tensorcore-exact/csrc/tc_emu3.cu (sha256 e2c27a15...ff6b), configuration
// cfg 6 (BM = BN = 128, WM = 2, WN = 4, KT = 32: 64 x 32 warp tiles, 256 threads).  Changes against the audited source:
//   - device-only: no torch/c10/cuda_fp16 headers, no TORCH_CHECKs, no host code; one extern "C" entry with the
//     standard ABI (the template kernel became a __device__ __forceinline__ body instantiated once);
//   - cuda_fp16.h replaced by the identical inline PTX of __hmul2 (mul.f16x2) and the constant 256.0 in fp16
//     (0x5C00), so that NVRTC needs no CUDA headers;
//   - the other configurations (cfg 0-5, 7) are not shipped.
// Arithmetic is unchanged.
//
// ABI (checked by the Python wrapper, never by this kernel): A [M][K] E4M3 codes row-major, B [N][K] row-major,
// C [M][N] fp32 row-major = Ada fast chain over k = 0..K-1 from +0; 1 <= M, N; 16 <= K < k_max (1 << 24, see
// docs/kernels_fast.md); K % 16 == 0; finite codes; 16-byte aligned A and B (16-byte vector loads);
// grid = (ceil(N/128), ceil(M/128), 1), block = (256, 1, 1), dynamic shared memory = 81,920 B (above the 48 KiB
// default: the loader must set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES and check the call; plus 1 KiB static).
// K == 0 writes +0 without reading A or B (every global load is guarded by gk < K).  The kernel does not fault for any
// byte value: the only data-dependent address is s_cpx[exponent field of X], X >= +0 by construction (sum of
// non-negative bf16 powers of two with c = +0), so the index is in [0, 255].
//
// Original description (p3 tc_emu3):
// tc_emu3: tc_emu2 plus (i) deferred accumulator truncation: the running value t is kept untruncated; Ada's
// truncation of each block result to 13 fraction bits is merged into the next block's truncation of c to the grid
// 2^(E'-13) (nested truncations toward zero on dyadic grids compose, and E' >= exponent of t), and applied once at
// the end; (ii) the exponent decode X -> bits of 2^(maxep+11) by a 256-entry shared-memory table.
// tc_emu2: optimised version of tc_emu (same numerics, see tc_emu.cu for the method).
// Changes: per-output block bookkeeping in the fp32 bit-pattern domain (fewer integer-pipe ops), 16-byte global loads
// prefetched into registers one K-chunk ahead, double-buffered shared memory (one barrier per chunk).
// Method (tc_emu, design A): per 16-term Ada block with block exponent E = max(max product exponent, exponent(acc)),
// each 8-term half is run twice through mma.m16n8k8 f16 with the dummy accumulator c = +2^(E+11) and c = -2^(E+11):
// the HMMA aligns the products to the exponent of c and truncates them to the grid 2^(E-13) (F = 24), so the half
// sum S*u is rp - cp when rp < cp (S < 0) and rn + cp otherwise (sign(S) != sign(c) keeps the result exact).
#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#else
#include <stdint.h>
#endif

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4b(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void hmma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0, const float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void bmma_k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}

__device__ __forceinline__ uint32_t pair_f16(uint32_t two) {  // bytes b0 | b1 << 8 (E4M3) -> fp16x2 (exact)
  const uint32_t r = ((two & 0x80u) << 8) | ((two & 0x7fu) << 7) | ((two & 0x8000u) << 16) | ((two & 0x7f00u) << 15);
  uint32_t h;  // = __hmul2(r, {256, 256}): the inline PTX of cuda_fp16.hpp (CUDA 12.8); exact (power-of-two scaling)
  asm("{mul.f16x2 %0,%1,%2;\n}" : "=r"(h) : "r"(r), "r"(0x5C005C00u));
  return h;
}
__device__ __forceinline__ uint32_t enc1(uint32_t b) {  // bf16 bits of 2^(5*ea) (ea clamped at -6), 0 for +-0
  const int e = (b >> 3) & 15;
  return (b & 0x7f) ? (uint32_t)((5 * max(e, 1) + 92) << 7) : 0u;
}
__device__ __forceinline__ void cvt4(uint32_t w, uint32_t& h01, uint32_t& h23, uint32_t& x01, uint32_t& x23) {
  h01 = pair_f16(w & 0xffffu); h23 = pair_f16(w >> 16);
  x01 = enc1(w & 0xff) | (enc1((w >> 8) & 0xff) << 16);
  x23 = enc1((w >> 16) & 0xff) | (enc1(w >> 24) << 16);
}

template <int BM, int BN, int WM, int WN, int KT>
struct Cfg {
  static constexpr int NW = WM * WN, NTH = NW * 32, TM = BM / WM, TN = BN / WN, MT = TM / 16, NT = TN / 8;
  static constexpr int LD = KT + 8;                  // 16-bit elements per smem row (+16 B: conflict-free ldmatrix)
  static constexpr int VPR = KT / 16;                // 16-byte global vectors per row per chunk
  static constexpr int VA = (BM * VPR + NTH - 1) / NTH, VB = (BN * VPR + NTH - 1) / NTH;
  static constexpr int BUF = (BM + BN) * LD * 2;     // 16-bit elements per buffer (values + exponent codes)
};

template <int BM, int BN, int WM, int WN, int KT>
__device__ __forceinline__ void tc3_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                         float* __restrict__ C, int M, int N, int K) {
  using G = Cfg<BM, BN, WM, WN, KT>;
  constexpr int LD = G::LD, MT = G::MT, NT = G::NT, TM = G::TM, TN = G::TN;
  extern __shared__ __align__(16) uint16_t smem[];
  __shared__ int s_cpx[256];   // biased exponent of X -> bits of 2^(maxep + 11); X == 0 -> 0 (never the maximum)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 256; i += G::NTH) {
    const int ex = i - 127;                                   // floor(log2 X) in [5 maxep, 5 maxep + 4]
    const int mx = (ex + 60 >= 0) ? (ex + 60) / 5 - 12 : -1000;
    s_cpx[i] = (i == 0 || mx < -12 || mx > 16) ? 0 : ((mx + 138) << 23);
  }
  const int wm = warp / WN, wn = warp % WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  uint4 ra[G::VA], rb[G::VB];
  auto gload = [&](int k0) {
#pragma unroll
    for (int v = 0; v < G::VA; ++v) {
      const int idx = tid + v * G::NTH, r = idx / G::VPR, c = idx % G::VPR, gm = m0 + r, gk = k0 + 16 * c;
      ra[v] = (idx < BM * G::VPR && gm < M && gk < K) ? __ldg(reinterpret_cast<const uint4*>(A + (size_t)gm * K + gk))
                                                      : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int v = 0; v < G::VB; ++v) {
      const int idx = tid + v * G::NTH, r = idx / G::VPR, c = idx % G::VPR, gn = n0 + r, gk = k0 + 16 * c;
      rb[v] = (idx < BN * G::VPR && gn < N && gk < K) ? __ldg(reinterpret_cast<const uint4*>(B + (size_t)gn * K + gk))
                                                      : make_uint4(0, 0, 0, 0);
    }
  };
  auto sstore = [&](int buf) {
    uint16_t* sV = smem + buf * G::BUF;       // [BM + BN][LD] fp16 values
    uint16_t* sX = sV + (BM + BN) * LD;       // [BM + BN][LD] bf16 exponent codes
    auto put = [&](int row, int c, const uint4& w) {
      uint32_t h[8], x[8];
      cvt4(w.x, h[0], h[1], x[0], x[1]); cvt4(w.y, h[2], h[3], x[2], x[3]);
      cvt4(w.z, h[4], h[5], x[4], x[5]); cvt4(w.w, h[6], h[7], x[6], x[7]);
      uint4* pv = reinterpret_cast<uint4*>(&sV[row * LD + 16 * c]);
      uint4* px = reinterpret_cast<uint4*>(&sX[row * LD + 16 * c]);
      pv[0] = make_uint4(h[0], h[1], h[2], h[3]); pv[1] = make_uint4(h[4], h[5], h[6], h[7]);
      px[0] = make_uint4(x[0], x[1], x[2], x[3]); px[1] = make_uint4(x[4], x[5], x[6], x[7]);
    };
#pragma unroll
    for (int v = 0; v < G::VA; ++v) {
      const int idx = tid + v * G::NTH;
      if (idx < BM * G::VPR) put(idx / G::VPR, idx % G::VPR, ra[v]);
    }
#pragma unroll
    for (int v = 0; v < G::VB; ++v) {
      const int idx = tid + v * G::NTH;
      if (idx < BN * G::VPR) put(BM + idx / G::VPR, idx % G::VPR, rb[v]);
    }
  };

  const int nchunk = (K + KT - 1) / KT;
  gload(0);
  sstore(0);
  __syncthreads();
  for (int ch = 0; ch < nchunk; ++ch) {
    const int k0 = ch * KT;
    if (ch + 1 < nchunk) gload(k0 + KT);
    const uint16_t* sV = smem + (ch & 1) * G::BUF;
    const uint16_t* sX = sV + (BM + BN) * LD;
    const int nblk = min(KT, K - k0) / 16;
    for (int g = 0; g < nblk; ++g) {
      const int kk = g * 16;
      uint32_t fa[MT][4], fax[MT][4], fb[NT][2], fbx[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int row = wm * TM + i * 16 + (lane & 15), col = kk + (lane >> 4) * 8;
        ldsm_x4(fa[i], &sV[row * LD + col]);
        ldsm_x4(fax[i], &sX[row * LD + col]);
      }
#pragma unroll
      for (int j = 0; j < NT; j += 2) {  // two n-tiles per ldmatrix.x4: (n 0-7, k lo), (n 0-7, k hi), (n 8-15, ..)
        const int row = BM + wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3), col = kk + ((lane >> 3) & 1) * 8;
        ldsm_x4b(fb[j][0], fb[j][1], fb[j + 1][0], fb[j + 1][1], &sV[row * LD + col]);
        ldsm_x4b(fbx[j][0], fbx[j][1], fbx[j + 1][0], fbx[j + 1][1], &sX[row * LD + col]);
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          float X[4], cp[4], cn[4];
          bmma_k16(X, fax[i], fbx[j][0], fbx[j][1]);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int cpx = s_cpx[__float_as_uint(X[q]) >> 23];
            const int cpa = (__float_as_int(acc[i][j][q]) & 0x7f800000) + (11 << 23);         // 2^(ec + 11)
            cp[q] = __int_as_float(max(cpx, cpa));
            cn[q] = -cp[q];
          }
          float r0p[4], r0n[4], r1p[4], r1n[4];
          hmma_k8(r0p, fa[i][0], fa[i][1], fb[j][0], cp);
          hmma_k8(r0n, fa[i][0], fa[i][1], fb[j][0], cn);
          hmma_k8(r1p, fa[i][2], fa[i][3], fb[j][1], cp);
          hmma_k8(r1n, fa[i][2], fa[i][3], fb[j][1], cn);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float s0 = r0p[q] < cp[q] ? __fsub_rn(r0p[q], cp[q]) : __fadd_rn(r0n[q], cp[q]);
            const float s1 = r1p[q] < cp[q] ? __fsub_rn(r1p[q], cp[q]) : __fadd_rn(r1n[q], cp[q]);
            const float a = acc[i][j][q];   // untruncated running value t of the previous block
            const float m = __int_as_float((__float_as_int(cp[q]) - (1 << 23)) | (__float_as_int(a) & 0x80000000));
            acc[i][j][q] = __fadd_rn(__fadd_rn(s0, s1), __fsub_rn(__fadd_rz(a, m), m));
          }
        }
    }
    if (ch + 1 < nchunk) sstore((ch + 1) & 1);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {   // Ada result: 13 fraction bits, toward zero
        const float t = acc[i][j][q];
        const float m2 = __int_as_float((__float_as_int(t) & 0xff800000) + (10 << 23));
        acc[i][j][q] = __fsub_rn(__fadd_rz(t, m2), m2);
      }
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int gm = m0 + wm * TM + i * 16 + gq, gn = n0 + wn * TN + j * 8 + 2 * tq;
      if (gm < M) {
        if (gn + 1 < N && ((N & 1) == 0)) *reinterpret_cast<float2*>(&C[(size_t)gm * N + gn]) = make_float2(acc[i][j][0], acc[i][j][1]);
        else { if (gn < N) C[(size_t)gm * N + gn] = acc[i][j][0]; if (gn + 1 < N) C[(size_t)gm * N + gn + 1] = acc[i][j][1]; }
      }
      if (gm + 8 < M) {
        if (gn + 1 < N && ((N & 1) == 0)) *reinterpret_cast<float2*>(&C[(size_t)(gm + 8) * N + gn]) = make_float2(acc[i][j][2], acc[i][j][3]);
        else { if (gn < N) C[(size_t)(gm + 8) * N + gn] = acc[i][j][2]; if (gn + 1 < N) C[(size_t)(gm + 8) * N + gn + 1] = acc[i][j][3]; }
      }
    }
}

}  // namespace

// Standard-ABI entry (KernelSpec "tc3": tile 128 x 128, 256 threads, smem_bytes 81,920, align 16, k_max 1 << 24).
extern "C" __global__ void __launch_bounds__(256)
mmaemu_tc3_e4m3e4m3(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
                    int M, int N, int K) {
  tc3_body<128, 128, 2, 4, 32>(A, B, C, M, N, K);
}
