// mmaemu v9: Ada FP8 (E4M3 x E4M3) fast-accumulation chain, kernel ABI v1 (mmaemu/backends/cuda/spec.py).
//
// Provenance: the tcemu audit's p2-int-mixed/v9.cu (sha256 c07a5f58...6f4), configuration cfg 1
// (WM = 2, WN = 2, MINB = 2, OPT = 2: 64 x 64 output tile, 128 threads, block_init_m / block_finish_m, the raw
// pbits64 exponent mapping).  Changes against the audited source:
//   - device-only: no torch/c10 headers, no TORCH_CHECKs, no host code; one extern "C" entry with the standard ABI
//     (the template kernel became a __device__ __forceinline__ body instantiated once);
//   - K = 0 guard: C = +0 without touching A or B (the audited kernel read A/B at k = 0 and faulted for K = 0);
//   - shared memory: 2 x sizeof(Tile8<64, 64>) = 36,864 B of dynamic shared memory (below the 48 KiB default, so no
//     opt-in attribute is needed); the loader checks it against the device limit instead of the old hard-coded
//     "<= 99 KB" check;
//   - the other configurations (cfg 0, 2-5) are not shipped.
// Arithmetic is unchanged: the block exponent comes from a bf16 tensor-core product of powers of two (the "E-pass",
// valid where the bf16 MMA has the window property -- checked per device by the "window" gate), the products are
// accumulated by an FFMA.RZ chain on the grid 2^(E-13) and a byte-SIMD correction turns floor into trunc.
//
// ABI (checked by the Python wrapper, never by this kernel): A [M][K] E4M3 codes row-major, B [N][K] row-major,
// C [M][N] fp32 row-major = Ada fast chain over k = 0..K-1 from +0; 1 <= M, N; 16 <= K < 2^24 (keeps the block
// exponent <= 55, the limit of the byte encodings); K % 16 == 0; finite codes; 4-byte aligned pointers;
// grid = (ceil(N/64), ceil(M/64), 1), block = (128, 1, 1), dynamic shared memory = 36,864 B.
// The kernel does not fault for any byte value (no address depends on the data).
//
// Original description (p2-int-mixed v8/v9):
// v4's algorithm with the block-exponent pass (E-pass) moved to the TENSOR cores.  The base-64 E-pass is a matrix
// product of powers of two: sum_k 2^(6 ea_ik - 7) * 2^(6 eb_jk - 7) (+ 2^-86).  It is computed with
// mma.sync.m16n8k16 bf16 x bf16 -> fp32.  Only the exponent of the result is used: the largest product is a power of
// two that the tensor core neither shifts nor truncates (it is the alignment reference), all other terms are >= 0,
// and there are <= 17 terms, so the result lies in [64^E, 17 * 64^E] whatever the internal alignment/truncation of
// the smaller terms, and its exponent field is 6E + 113 + d, d in [0, 4] (as in the CUDA-core bf16x2 E-pass).
// Each warp owns a 32x32 output tile laid out as the mma accumulator fragments: lane (g = lane/4, t = lane%4) holds
// rows g + 8i (i = 0..3) and columns 2t + 8n + p (n = 0..3, p = 0..1); the value tiles in shared memory are stored in
// a permuted order so that these rows / columns are contiguous (one LDS.128 for the 4 rows, two for the 8 columns).
#include "v9_common.cuh"

namespace {
using namespace mmaemu_v9;

constexpr int ESTRIDE = 24;  // uint16 per row of the E-operand tiles (16 used; 48-byte stride: conflict-free ldmatrix)

template <int BM, int BN>
struct Tile8 {
  float Av[16][BM];            // row slot = (r & ~31) | ((r & 7) << 2) | ((r >> 3) & 3)
  float Bv[16][BN];            // col slot = (c & ~31) | (((c >> 1) & 3) << 3) | (((c >> 3) & 3) << 1) | (c & 1)
  uint16_t AE[BM][ESTRIDE];    // bf16 2^(6 ea - 7) (byte-placed), [row][k]
  uint16_t BE[BN][ESTRIDE];    // [col][k]
  uint32_t Al[BM][4], As[BM][4];
  uint32_t Bl[BN][4], Bs[BN][4];
};

__device__ __forceinline__ int slot_a(int r) { return (r & ~31) | ((r & 7) << 2) | ((r >> 3) & 3); }
__device__ __forceinline__ int slot_b(int c) { return (c & ~31) | (((c >> 1) & 3) << 3) | (((c >> 3) & 3) << 1) | (c & 1); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float block_init_m(float V, uint32_t pb_raw, float& Mm_out) {
  const uint32_t e2 = __float_as_uint(fmaxf(fabsf(V), __uint_as_float(pb_raw))) & 0x7f800000u;
  const uint32_t mm = e2 + 0x05400000u;
  const float mc = __uint_as_float((__float_as_uint(V) & 0x80000000u) | mm);
  const float ct = __fsub_rn(__fadd_rz(V, mc), mc);
  Mm_out = __uint_as_float(mm);
  return __fadd_rn(ct, Mm_out);
}
__device__ __forceinline__ float block_finish_m(float S, float Mm, const uint32_t (&la)[4], const uint32_t (&sa)[4],
                                                const uint32_t (&lb)[4], const uint32_t (&sb)[4]) {
  const uint32_t s = __float_as_uint(S);
  const uint32_t tt = (260u - (s >> 23)) * 0x01010101u;
  uint32_t corr = 0;
#pragma unroll
  for (int q = 0; q < 4; ++q) corr = dp4a_u(lop3_corr(la[q] + lb[q] + tt, sa[q], sb[q]), 0x01010101u, corr);
  return __fsub_rn(__uint_as_float(s + (corr >> 7)), Mm);
}

// cfg 1 body (WM = WN = 2, OPT = 2).  The template parameters are kept so that the code stays a line-by-line copy of
// the audited kernel; only <2, 2> is instantiated.
template <int WM, int WN>
__device__ __forceinline__ void v9_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                        float* __restrict__ C, int M, int N, int K) {
  constexpr int BM = 32 * WM, BN = 32 * WN, NT = WM * WN * 32;
  constexpr int WA = BM * 4 / NT, WB = BN * 4 / NT;
  static_assert(BM * 4 % NT == 0 && BN * 4 % NT == 0, "loader mapping");
  using T = Tile8<BM, BN>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* tiles = reinterpret_cast<T*>(smem_raw);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int nb = K / 16;
  // K = 0 guard (also any K < 16): with no 16-block, the load bounds Mg / Ng are 0, so the prologue loads nothing
  // (zero words), the block loop runs 0 times and the final store writes final_trunc(+0) = +0; A and B are never
  // read.  The audited kernel loaded block 0 unconditionally and faulted for K = 0.  Guarding through the load bounds
  // keeps the audited code shape (220 vs 216 registers on sm_89; 1.547 vs 1.549 TMAC/s at 1B M = 2048 in an
  // interleaved A/B run, where predicating the whole prologue instead cost 2%: 1.52 TMAC/s).
  const int Mg = nb > 0 ? M : 0, Ng = nb > 0 ? N : 0;

  uint32_t ra[WA], rb[WB];
  auto gload = [&](int blk) {
    const int k0 = 16 * blk;
#pragma unroll
    for (int u = 0; u < WA; ++u) {
      const int idx = tid + u * NT, r = idx % BM, w = idx / BM, gm = m0 + r;
      ra[u] = gm < Mg ? __ldg(reinterpret_cast<const uint32_t*>(A + (size_t)gm * K + k0 + 4 * w)) : 0u;
    }
#pragma unroll
    for (int u = 0; u < WB; ++u) {
      const int idx = tid + u * NT, r = idx % BN, w = idx / BN, gn = n0 + r;
      rb[u] = gn < Ng ? __ldg(reinterpret_cast<const uint32_t*>(B + (size_t)gn * K + k0 + 4 * w)) : 0u;
    }
  };
  auto sstore = [&](T& s) {
#pragma unroll
    for (int u = 0; u < WA; ++u) {
      const int idx = tid + u * NT, r = idx % BM, w = idx / BM, sr = slot_a(r);
      uint32_t sg, la, x; float v[4];
      decode_word(ra[u], sg, la, x, v);
      s.Av[4 * w + 0][sr] = v[0]; s.Av[4 * w + 1][sr] = v[1]; s.Av[4 * w + 2][sr] = v[2]; s.Av[4 * w + 3][sr] = v[3];
      // 4 bf16 values [0,x0] [0,x1] [0,x2] [0,x3] -> two 32-bit words
      *reinterpret_cast<uint2*>(&s.AE[r][4 * w]) = make_uint2(__byte_perm(x, 0u, 0x1404u), __byte_perm(x, 0u, 0x3424u));
      s.Al[r][w] = la; s.As[r][w] = sg;
    }
#pragma unroll
    for (int u = 0; u < WB; ++u) {
      const int idx = tid + u * NT, r = idx % BN, w = idx / BN, sc = slot_b(r);
      uint32_t sg, la, x; float v[4];
      decode_word(rb[u], sg, la, x, v);
      s.Bv[4 * w + 0][sc] = v[0]; s.Bv[4 * w + 1][sc] = v[1]; s.Bv[4 * w + 2][sc] = v[2]; s.Bv[4 * w + 3][sc] = v[3];
      *reinterpret_cast<uint2*>(&s.BE[r][4 * w]) = make_uint2(__byte_perm(x, 0u, 0x1404u), __byte_perm(x, 0u, 0x3424u));
      s.Bl[r][w] = la; s.Bs[r][w] = sg;
    }
  };

  float acc[4][8];
  float Mk[4][8];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

  gload(0);
  int st = 0;
  sstore(tiles[0]);
  __syncthreads();
#pragma unroll 1
  for (int b = 0; b < nb; ++b) {
    const T& s = tiles[st];
    if (b + 1 < nb) gload(b + 1);
    // ---- (1) E-pass on the tensor cores: 2 x 4 mma.m16n8k16 per warp ----
    uint32_t Pb[4][8];
    {
      uint32_t af[2][4], bf[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        ldsm_x4(af[mt], &s.AE[wm * 32 + 16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1)][8 * (lane >> 4)]);
#pragma unroll
      for (int np = 0; np < 2; ++np)  // n-tiles 2np, 2np+1: matrices (n 0-7,k 0-7) (n 0-7,k 8-15) (n 8-15,k 0-7) (n 8-15,k 8-15)
        ldsm_x4(bf[np], &s.BE[wn * 32 + 16 * np + (lane & 7) + 8 * (lane >> 4)][8 * ((lane >> 3) & 1)]);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          float d[4] = {0x1p-86f, 0x1p-86f, 0x1p-86f, 0x1p-86f};
          mma_bf16(d, af[mt], bf[nt >> 1][2 * (nt & 1)], bf[nt >> 1][2 * (nt & 1) + 1]);
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int p = 0; p < 2; ++p) Pb[2 * mt + h][2 * nt + p] = pbits64_raw(__float_as_uint(d[2 * h + p]) >> 23);
        }
    }
    // ---- (2) block init ----
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[i][j] = block_init_m(acc[i][j], Pb[i][j], Mk[i][j]);
    // ---- (3) FFMA.RZ chain ----
#pragma unroll
    for (int c = 0; c < 16; ++c) {
      const float4 a4 = *reinterpret_cast<const float4*>(&s.Av[c][wm * 32 + 4 * g]);
      const float4 b4a = *reinterpret_cast<const float4*>(&s.Bv[c][wn * 32 + 8 * t]);
      const float4 b4b = *reinterpret_cast<const float4*>(&s.Bv[c][wn * 32 + 8 * t + 4]);
      const float va[4] = {a4.x, a4.y, a4.z, a4.w};
      const float vb[8] = {b4a.x, b4a.y, b4a.z, b4a.w, b4b.x, b4b.y, b4b.z, b4b.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rz(va[i], vb[j], acc[i][j]);
    }
    // ---- (4) correction, V = S - M ----
    {
      uint32_t lb[8][4], sb[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = wn * 32 + 2 * t + 8 * (j >> 1) + (j & 1);
        const uint4 l = *reinterpret_cast<const uint4*>(&s.Bl[col][0]);
        const uint4 q = *reinterpret_cast<const uint4*>(&s.Bs[col][0]);
        lb[j][0] = l.x; lb[j][1] = l.y; lb[j][2] = l.z; lb[j][3] = l.w;
        sb[j][0] = q.x; sb[j][1] = q.y; sb[j][2] = q.z; sb[j][3] = q.w;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = wm * 32 + g + 8 * i;
        const uint4 l = *reinterpret_cast<const uint4*>(&s.Al[row][0]);
        const uint4 q = *reinterpret_cast<const uint4*>(&s.As[row][0]);
        const uint32_t la[4] = {l.x, l.y, l.z, l.w}, sa[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = block_finish_m(acc[i][j], Mk[i][j], la, sa, lb[j], sb[j]);
      }
    }
    if (b + 1 < nb) sstore(tiles[st ^ 1]);
    __syncthreads();
    st ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int gm = m0 + wm * 32 + g + 8 * i, gn = n0 + wn * 32 + 2 * t + 8 * (j >> 1) + (j & 1);
      if (gm < M && gn < N) C[(size_t)gm * N + gn] = final_trunc(acc[i][j]);
    }
}

}  // namespace

// Standard-ABI entry (KernelSpec "v9": tile 64 x 64, 128 threads, smem_bytes 36,864, align 4, k_max 1 << 24).
extern "C" __global__ void __launch_bounds__(128, 2)
mmaemu_v9_e4m3e4m3(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
                   int M, int N, int K) {
  v9_body<2, 2>(A, B, C, M, N, K);
}
