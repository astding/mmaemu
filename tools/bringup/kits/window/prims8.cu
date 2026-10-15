// Exhaustive test hook for v8's tensor-core E-pass: one warp per 16x8 output tile; A rows / B columns are 16 E4M3
// bytes each (one 16-term block); returns fp32 bits of 2^E_prod (exponent only) for all 128 (row, col) pairs, computed
// exactly as in v8_kernel (decode_word -> byte-placed bf16 -> ldmatrix -> mma.sync m16n8k16 bf16 -> pbits64_raw).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "v2_common.cuh"
using namespace p2;

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__global__ void k_epass_mma(const uint8_t* A, const uint8_t* B, int32_t* out) {
  __shared__ __align__(16) uint16_t AE[16][24];
  __shared__ __align__(16) uint16_t BE[16][24];
  const int lane = threadIdx.x, tile = blockIdx.x;
  // lanes 0..15: A rows, 16..23: B rows (8 columns), 24..31: zero B rows 8..15 (unused second n-tile)
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
  ldsm_x4(af, &AE[(lane & 7) + 8 * ((lane >> 3) & 1)][8 * (lane >> 4)]);
  ldsm_x4(bf, &BE[(lane & 7) + 8 * (lane >> 4)][8 * ((lane >> 3) & 1)]);
  float d[4] = {0x1p-86f, 0x1p-86f, 0x1p-86f, 0x1p-86f};
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
  const int g = lane >> 2, t = lane & 3;
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p)
      out[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p] = pbits64_raw(__float_as_uint(d[2 * h + p]) >> 23) & 0x7f800000u;
}

torch::Tensor epass_mma(torch::Tensor a, torch::Tensor b) {  // a [T*16, 16] uint8, b [T*8, 16] uint8
  const int T = a.size(0) / 16;
  auto out = torch::empty({T, 16, 8}, a.options().dtype(torch::kInt32));
  k_epass_mma<<<T, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), out.data_ptr<int32_t>());
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("epass_mma", &epass_mma); }
