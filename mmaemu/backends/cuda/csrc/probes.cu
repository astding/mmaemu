// mmaemu probes: instruction-level probes of the fp16 / bf16 tensor-core MMA numerics that the gated kernels rely on.
// They are shipped in the fatbin for sm_80, 86, 89, 90, 100 and 120 and built with the same flags as the kernels, so
// the per-device gates (mmaemu/backends/cuda/fast/gates.py) test the same instructions on every GPU -- and can reject
// a GPU where tc3 / v9 would be inexact.  Device-only: no CUDA/torch headers, no host code.
//
// Entries (all one warp per tile; launch grid = (T, 1, 1), block = (32, 1, 1), no dynamic shared memory):
//
//   mmaemu_probe_hmma_f16_k8(const uint16_t* A, const uint16_t* B, const float* C, float* D)
//       design-A probe: one mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 per FULL 16 x 8 tile.
//       A [T][16][8] fp16 bits (row r, k), B [T][8][8] fp16 bits (column n, k), C / D [T][16][8] fp32 (row r, col n):
//       D[r][n] = HMMA(sum_k A[r][k] * B[n][k] + C[r][n]).  Every output of the tile carries its own test vector.
//       (Full-tile version of the row-0 probe of the tcemu audit's p3-tensorcore-exact/csrc/tc_probe.cu,
//       sha256 41169964...739a; the fragment mapping is the PTX-ISA m16n8k8 layout used there and in tc3.cu.)
//
//   mmaemu_probe_epass_raw(const uint8_t* A, const uint8_t* B, int* D)
//       v9 E-pass probe: the exact v9 data path (decode_word -> byte-placed bf16 2^(6ea-7) -> ldmatrix ->
//       mma.sync.m16n8k16 bf16 with c = 2^-86) on E4M3 bytes; returns the RAW fp32 bits of the MMA result.
//       A [T*16][16] E4M3 codes (4-byte aligned), B [T*8][16] (row n = column n of the tile), D [T][16][8] int32.
//
//   mmaemu_probe_mma_bf16_raw(const uint16_t* A, const uint16_t* B, const float* C, int* D)
//       the same mma.sync.m16n8k16 bf16 on arbitrary bf16 bit patterns: A [T*16][16], B [T*8][16], C [T][16][8] fp32,
//       D [T][16][8] int32 raw fp32 bits.  Used for the base-32 exponent pass of tc3 (c = +0).
//
// Provenance of the last two: the tcemu audit's verify-p2-int-mixed-3/epass_raw.cu (sha256
// 1a8d205b...37e0), k_epass_raw and k_mma_raw, unchanged except for the extern "C" names and the removal of the torch
// host wrappers.
#include "v9_common.cuh"

namespace {
using namespace mmaemu_v9;

__device__ __forceinline__ void probe_ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void probe_mma_bf16(float (&d)[4], const uint32_t (&af)[4], const uint32_t (&bf)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
}

}  // namespace

extern "C" __global__ void __launch_bounds__(32)
mmaemu_probe_hmma_f16_k8(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, const float* __restrict__ C,
                         float* __restrict__ D) {
  const int v = blockIdx.x, lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const uint16_t* a = A + (size_t)v * 128;   // [16][8]
  const uint16_t* b = B + (size_t)v * 64;    // [8 n][8 k]
  const float* c = C + (size_t)v * 128;      // [16][8]
  const uint32_t a0 = (uint32_t)a[g * 8 + 2 * t] | ((uint32_t)a[g * 8 + 2 * t + 1] << 16);
  const uint32_t a1 = (uint32_t)a[(g + 8) * 8 + 2 * t] | ((uint32_t)a[(g + 8) * 8 + 2 * t + 1] << 16);
  const uint32_t b0 = (uint32_t)b[g * 8 + 2 * t] | ((uint32_t)b[g * 8 + 2 * t + 1] << 16);   // B[k=2t..2t+1][n=g]
  float d0 = c[g * 8 + 2 * t], d1 = c[g * 8 + 2 * t + 1], d2 = c[(g + 8) * 8 + 2 * t], d3 = c[(g + 8) * 8 + 2 * t + 1];
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(b0));
  float* dd = D + (size_t)v * 128;
  dd[g * 8 + 2 * t] = d0; dd[g * 8 + 2 * t + 1] = d1; dd[(g + 8) * 8 + 2 * t] = d2; dd[(g + 8) * 8 + 2 * t + 1] = d3;
}

extern "C" __global__ void __launch_bounds__(32)
mmaemu_probe_epass_raw(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, int32_t* __restrict__ out) {
  __shared__ __align__(16) uint16_t AE[16][24];
  __shared__ __align__(16) uint16_t BE[16][24];
  const int lane = threadIdx.x, tile = blockIdx.x;
  for (int w = 0; w < 4; ++w) {
    uint32_t word = 0;
    if (lane < 16) word = *reinterpret_cast<const uint32_t*>(A + (size_t)(tile * 16 + lane) * 16 + 4 * w);
    else if (lane < 24) word = *reinterpret_cast<const uint32_t*>(B + (size_t)(tile * 8 + lane - 16) * 16 + 4 * w);
    uint32_t sg, la, x; float v[4];
    decode_word(word, sg, la, x, v);
    const uint2 e = make_uint2(__byte_perm(x, 0u, 0x1404u), __byte_perm(x, 0u, 0x3424u));
    if (lane < 16) *reinterpret_cast<uint2*>(&AE[lane][4 * w]) = e;
    else *reinterpret_cast<uint2*>(&BE[lane - 16][4 * w]) = e;
  }
  __syncwarp();
  uint32_t af[4], bf[4];
  probe_ldsm_x4(af, &AE[(lane & 7) + 8 * ((lane >> 3) & 1)][8 * (lane >> 4)]);
  probe_ldsm_x4(bf, &BE[(lane & 7) + 8 * (lane >> 4)][8 * ((lane >> 3) & 1)]);
  float d[4] = {0x1p-86f, 0x1p-86f, 0x1p-86f, 0x1p-86f};
  probe_mma_bf16(d, af, bf);
  const int g = lane >> 2, t = lane & 3;
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p) out[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p] = __float_as_int(d[2 * h + p]);
}

extern "C" __global__ void __launch_bounds__(32)
mmaemu_probe_mma_bf16_raw(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, const float* __restrict__ C,
                          int32_t* __restrict__ out) {
  __shared__ __align__(16) uint16_t AE[16][24];
  __shared__ __align__(16) uint16_t BE[16][24];
  const int lane = threadIdx.x, tile = blockIdx.x;
  for (int k = 0; k < 16; ++k) {
    if (lane < 16) AE[lane][k] = A[(size_t)(tile * 16 + lane) * 16 + k];
    else if (lane < 24) BE[lane - 16][k] = B[(size_t)(tile * 8 + lane - 16) * 16 + k];
    else BE[lane - 16][k] = 0;
  }
  __syncwarp();
  uint32_t af[4], bf[4];
  probe_ldsm_x4(af, &AE[(lane & 7) + 8 * ((lane >> 3) & 1)][8 * (lane >> 4)]);
  probe_ldsm_x4(bf, &BE[(lane & 7) + 8 * (lane >> 4)][8 * ((lane >> 3) & 1)]);
  const int g = lane >> 2, t = lane & 3;
  float d[4];
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p) d[2 * h + p] = C[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p];
  probe_mma_bf16(d, af, bf);
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p) out[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p] = __float_as_int(d[2 * h + p]);
}
