// [verify-p3-tensorcore-exact-0] COPY of p3-tensorcore-exact/csrc/tc_emuB3.cu with ONE change: every mma.sync
// can be replaced (template MODE) by a warp-level SOFTWARE m16n8k16 bf16 MMA that implements a parametrised
// block-FMA model (exact, in double precision), using the PTX-ISA fragment layout:
//   MODE 0: hardware mma.sync (unchanged kernel)
//   MODE 1: software model of the Ada/Ampere bf16 HMMA: two chained 8-term blocks, c in the alignment, F=24, trunc
//   MODE 2: software model of the B200 (= RTX PRO 6000 per the paper) bf16 HMMA: one 16-term block, F=25, trunc
//   MODE 3: hypothetical sm_120 variant: one 16-term block, F=25, but c added LATE (fp32 RNE after normalisation)
//   MODE 4: hypothetical sm_120 variant: one 16-term block, F=26, c in the alignment
// MODE 1 on the RTX 4060 must equal MODE 0 bit for bit (validates the software MMA incl. its fragment layout);
// MODE 2 vs tcemu.reference.gemm then tests the KERNEL's implementation of design B under the paper's model.
// tc_emuB3: design B (Hopper / Blackwell incl. sm_120 bf16 HMMA: 16-term blocks, c in the alignment, 25 fraction bits)
// with the tc_emu3 bookkeeping. Timing proxy on Ada/Ampere (NOT bit-exact there); numerics checked in software
// (scripts/sim_designs.py, design 'Bb'). Per Ada block, half h: one m16n8k16 bf16 HMMA, k-slots 0-7 = real
// products, k-slot 8+j = dummy -2^(E+12) for output column j (A registers a4..a7 of lane (g,t) are exactly the lane's
// own outputs), c = +2^(E+12): alignment exponent E+12 -> products truncated to the Ada grid 2^(E-13); c and the dummy
// cancel; the result is exactly the truncated half sum.
// (tc_emu3 header:) tc_emu3: tc_emu2 plus (i) deferred accumulator truncation: the running value t is kept untruncated; Ada's
// truncation of each block result to 13 fraction bits is merged into the next block's truncation of c to the grid
// 2^(E'-13) (nested truncations toward zero on dyadic grids compose, and E' >= exponent of t), and applied once at
// the end; (ii) the exponent decode X -> bits of 2^(maxep+11) by a 256-entry shared-memory table.
// tc_emu2: optimised version of tc_emu (same numerics, see tc_emu.cu for the method).
// Changes: per-output block bookkeeping in the fp32 bit-pattern domain (fewer integer-pipe ops), 16-byte global loads
// prefetched into registers one K-chunk ahead, double-buffered shared memory (one barrier per chunk).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

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
__device__ __forceinline__ void hmma_k16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1, const float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void bmma_k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}


// ---------------- software m16n8k16 bf16 MMA (verify-p3-tensorcore-exact-0) ----------------
__device__ __forceinline__ float bf_lo(uint32_t r) { return __uint_as_float(r << 16); }
__device__ __forceinline__ float bf_hi(uint32_t r) { return __uint_as_float(r & 0xffff0000u); }
__device__ __forceinline__ int ilog_clamped(float x, int emin) { return max(ilogbf(x), emin); }
// one block FMA: n products + c; LATE: c added afterwards with fp32 RNE
template <int F, bool LATE>
__device__ __noinline__ float soft_block(const float* a, const float* b, int n, float c) {
  int E = -100000;
#pragma unroll 1
  for (int i = 0; i < n; ++i)
    if (a[i] != 0.f && b[i] != 0.f) E = max(E, ilog_clamped(a[i], -126) + ilog_clamped(b[i], -126));
  if (!LATE && c != 0.f) E = max(E, ilogbf(c));
  E = max(E, -133);
  double s = 0.0;
#pragma unroll 1
  for (int i = 0; i < n; ++i) s += trunc(scalbn((double)a[i] * (double)b[i], F - E));
  if (!LATE) s += trunc(scalbn((double)c, F - E));
  double d = scalbn(s, E - F);
  if (d != 0.0) {
    const int ed = max(ilogb(d), -126);
    d = scalbn(trunc(scalbn(d, 23 - ed)), ed - 23);
  } else {
    d = 0.0;
  }
  float r = (float)d;
  if (LATE) r = __fadd_rn(r, c);
  return r;
}
template <int MODE>
__device__ float soft_dot(const float* a, const float* b, float c) {
  if (MODE == 1) { const float t = soft_block<24, false>(a, b, 8, c); return soft_block<24, false>(a + 8, b + 8, 8, t); }
  if (MODE == 2) return soft_block<25, false>(a, b, 16, c);
  if (MODE == 3) return soft_block<25, true>(a, b, 16, c);
  return soft_block<26, false>(a, b, 16, c);
}
// PTX ISA m16n8k16 (.bf16) fragments, g = lane >> 2, t = lane & 3:
//   a0 = A[g][2t,2t+1], a1 = A[g+8][2t,2t+1], a2 = A[g][2t+8,2t+9], a3 = A[g+8][2t+8,2t+9]   (low half = lower k)
//   b0 = B[2t,2t+1][g], b1 = B[2t+8,2t+9][g];   c0,c1 = C[g][2t,2t+1], c2,c3 = C[g+8][2t,2t+1]
template <int MODE>
__device__ __noinline__ void soft_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
                         const float (&c)[4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float Ar[2][16], Bc[2][16];
#pragma unroll
  for (int tt = 0; tt < 4; ++tt) {
    const int sa = 4 * g + tt;
    const uint32_t r0 = __shfl_sync(0xffffffffu, a0, sa), r1 = __shfl_sync(0xffffffffu, a1, sa);
    const uint32_t r2 = __shfl_sync(0xffffffffu, a2, sa), r3 = __shfl_sync(0xffffffffu, a3, sa);
    Ar[0][2 * tt] = bf_lo(r0); Ar[0][2 * tt + 1] = bf_hi(r0); Ar[1][2 * tt] = bf_lo(r1); Ar[1][2 * tt + 1] = bf_hi(r1);
    Ar[0][2 * tt + 8] = bf_lo(r2); Ar[0][2 * tt + 9] = bf_hi(r2); Ar[1][2 * tt + 8] = bf_lo(r3); Ar[1][2 * tt + 9] = bf_hi(r3);
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int sb = 4 * (2 * t + cc) + tt;
      const uint32_t q0 = __shfl_sync(0xffffffffu, b0, sb), q1 = __shfl_sync(0xffffffffu, b1, sb);
      Bc[cc][2 * tt] = bf_lo(q0); Bc[cc][2 * tt + 1] = bf_hi(q0); Bc[cc][2 * tt + 8] = bf_lo(q1); Bc[cc][2 * tt + 9] = bf_hi(q1);
    }
  }
  float out[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) out[q] = soft_dot<MODE>(Ar[q >> 1], Bc[q & 1], c[q]);
#pragma unroll
  for (int q = 0; q < 4; ++q) d[q] = out[q];
}
template <int MODE>
__device__ __forceinline__ void mma_any(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, const float (&c)[4]) {
  if (MODE == 0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    soft_mma<MODE>(d, a0, a1, a2, a3, b0, b1, c);
  }
}
// instruction probe: one tile per warp; D_mode = A[16x16] B[16x8] + C, inputs given as full tiles
template <int MODE>
__global__ void probe_tiles(const uint16_t* A, const uint16_t* B, const float* C, float* D) {
  const int v = blockIdx.x, lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const uint16_t* a = A + (size_t)v * 256; const uint16_t* b = B + (size_t)v * 128; const float* cc = C + (size_t)v * 128;
  auto av = [&](int r, int k) -> uint32_t { return a[r * 16 + k]; };
  auto bv = [&](int k, int n) -> uint32_t { return b[k * 8 + n]; };
  const uint32_t a0 = av(g, 2 * t) | (av(g, 2 * t + 1) << 16), a1 = av(g + 8, 2 * t) | (av(g + 8, 2 * t + 1) << 16);
  const uint32_t a2 = av(g, 2 * t + 8) | (av(g, 2 * t + 9) << 16), a3 = av(g + 8, 2 * t + 8) | (av(g + 8, 2 * t + 9) << 16);
  const uint32_t b0 = bv(2 * t, g) | (bv(2 * t + 1, g) << 16), b1 = bv(2 * t + 8, g) | (bv(2 * t + 9, g) << 16);
  const float c[4] = {cc[g * 8 + 2 * t], cc[g * 8 + 2 * t + 1], cc[(g + 8) * 8 + 2 * t], cc[(g + 8) * 8 + 2 * t + 1]};
  float d[4];
  mma_any<MODE>(d, a0, a1, a2, a3, b0, b1, c);
  float* dd = D + (size_t)v * 128;
  dd[g * 8 + 2 * t] = d[0]; dd[g * 8 + 2 * t + 1] = d[1]; dd[(g + 8) * 8 + 2 * t] = d[2]; dd[(g + 8) * 8 + 2 * t + 1] = d[3];
}

__device__ __forceinline__ uint32_t bf16_of(uint32_t b) {  // E4M3 byte -> bf16 bits (exact; subnormals via fp32 scaling)
  const float f = __int_as_float((int)((b & 0x7fu) << 20)) * 1.329227995784916e36f;  // * 2^120 (exact)
  return ((__float_as_uint(f) >> 16) & 0x7fffu) | ((b & 0x80u) << 8);
}
__device__ __forceinline__ uint32_t pair_f16(uint32_t two) {  // (name kept) bytes b0 | b1 << 8 -> bf16x2
  return bf16_of(two & 0xffu) | (bf16_of((two >> 8) & 0xffu) << 16);
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

template <int BM, int BN, int WM, int WN, int KT, int MODE>
__global__ void __launch_bounds__(WM * WN * 32)
tc_emuB3_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  using G = Cfg<BM, BN, WM, WN, KT>;
  constexpr int LD = G::LD, MT = G::MT, NT = G::NT, TM = G::TM, TN = G::TN;
  extern __shared__ __align__(16) uint16_t smem[];
  __shared__ int s_cpx[256];   // biased exponent of X -> bits of 2^(maxep + 11); X == 0 -> 0 (never the maximum)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 256; i += G::NTH) {
    const int ex = i - 127;                                   // floor(log2 X) in [5 maxep, 5 maxep + 4]
    const int mx = (ex + 60 >= 0) ? (ex + 60) / 5 - 12 : -1000;
    s_cpx[i] = (i == 0 || mx < -12 || mx > 16) ? 0 : ((mx + 139) << 23);   // 2^(maxep + 12)
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
          float X[4], cd[4];
          { const float z4[4] = {0.f, 0.f, 0.f, 0.f};
            mma_any<MODE>(X, fax[i][0], fax[i][1], fax[i][2], fax[i][3], fbx[j][0], fbx[j][1], z4); }
          uint32_t dum[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int cpx = s_cpx[__float_as_uint(X[q]) >> 23];
            const int cpa = (__float_as_int(acc[i][j][q]) & 0x7f800000) + (12 << 23);         // 2^(ec + 12)
            const int c = max(cpx, cpa);                                                     // 2^(E + 12)
            cd[q] = __int_as_float(c);
            dum[q] = ((uint32_t)c >> 16) | 0x8000u;                                          // bf16 -2^(E+12)
          }
          const uint32_t d01 = __byte_perm(dum[0], dum[1], 0x5410), d23 = __byte_perm(dum[2], dum[3], 0x5410);
          const int gq = lane >> 2, tq = lane & 3;
          const uint32_t bd = (gq == 2 * tq ? 0x3f80u : 0u) | (gq == 2 * tq + 1 ? 0x3f800000u : 0u);  // 1.0 at k = 8 + n
          float r0[4], r1[4];
          mma_any<MODE>(r0, fa[i][0], fa[i][1], d01, d23, fb[j][0], bd, cd);
          mma_any<MODE>(r1, fa[i][2], fa[i][3], d01, d23, fb[j][1], bd, cd);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float a = acc[i][j][q];
            const float m = __int_as_float((__float_as_int(cd[q]) - (2 << 23)) | (__float_as_int(a) & 0x80000000));
            acc[i][j][q] = __fadd_rn(__fadd_rn(r0[q], r1[q]), __fsub_rn(__fadd_rz(a, m), m));
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

template <int BM, int BN, int WM, int WN, int KT, int MODE>
torch::Tensor launch(torch::Tensor A, torch::Tensor B) {
  using G = Cfg<BM, BN, WM, WN, KT>;
  const int M = A.size(0), N = B.size(0), K = A.size(1);
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
  const int smem = 2 * G::BUF * 2;
  auto kern = tc_emuB3_kernel<BM, BN, WM, WN, KT, MODE>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  kern<<<grid, G::NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(),
                                                                   C.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

}  // namespace

template <int MODE>
torch::Tensor dispatch_cfg(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  switch (cfg) {
    case 0: return launch<64, 64, 2, 2, 32, MODE>(A, B);
    case 4: return launch<64, 64, 2, 2, 16, MODE>(A, B);
    case 6: return launch<128, 128, 2, 4, 32, MODE>(A, B);   // 64 x 32 warp tiles
    case 7: return launch<128, 64, 4, 2, 16, MODE>(A, B);
    default: TORCH_CHECK(false, "cfg must be 0, 4, 6 or 7 in this build");
  }
  return torch::Tensor();
}
torch::Tensor b3_gemm(torch::Tensor A, torch::Tensor B, int64_t mode, int64_t cfg) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.scalar_type() == torch::kUInt8 && B.scalar_type() == torch::kUInt8);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1) && A.size(1) % 16 == 0, "K % 16 == 0");
  A = A.contiguous(); B = B.contiguous();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(A.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(B.data_ptr()) % 16 == 0,
              "16-byte aligned inputs required");
  switch (mode) {
    case 0: return dispatch_cfg<0>(A, B, cfg);
    case 1: return dispatch_cfg<1>(A, B, cfg);
    case 2: return dispatch_cfg<2>(A, B, cfg);
    case 3: return dispatch_cfg<3>(A, B, cfg);
    default: return dispatch_cfg<4>(A, B, cfg);
  }
}
torch::Tensor probe(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t mode) {
  const int V = A.size(0);
  auto D = torch::empty({V, 16, 8}, C.options());
  auto st = c10::cuda::getCurrentCUDAStream();
  const uint16_t* a = (const uint16_t*)A.data_ptr<int16_t>(); const uint16_t* b = (const uint16_t*)B.data_ptr<int16_t>();
  switch (mode) {
    case 0: probe_tiles<0><<<V, 32, 0, st>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>()); break;
    case 1: probe_tiles<1><<<V, 32, 0, st>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>()); break;
    case 2: probe_tiles<2><<<V, 32, 0, st>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>()); break;
    case 3: probe_tiles<3><<<V, 32, 0, st>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>()); break;
    default: probe_tiles<4><<<V, 32, 0, st>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>()); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b3_gemm", &b3_gemm);
  m.def("probe", &probe);
}
