// verify-p2-int-mixed-3: raw-output probes of the bf16 tensor-core MMA used by v9's E-pass.
// (1) epass_raw(a, b): identical data path to prims8.cu / v9 (decode_word -> byte-placed bf16 -> ldmatrix ->
//     mma.sync.m16n8k16 bf16 -> fp32, c = 2^-86), but returns the RAW fp32 bits of d (not only the mapped exponent),
//     so the hardware result can be compared with the exact sum and with tensor-core models.
// (2) mma_bf16_raw(a16, b16, c): the same mma.sync.m16n8k16 on arbitrary bf16 bit patterns (a16 [T*16,16] row k,
//     b16 [T*8,16] col k, c [T,16,8] fp32), returning raw fp32 d: used to validate the model on the hardware.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "v2_common.cuh"
using namespace p2;

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&af)[4], const uint32_t (&bf)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
}

__global__ void k_epass_raw(const uint8_t* A, const uint8_t* B, int32_t* out) {
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
  ldsm_x4(af, &AE[(lane & 7) + 8 * ((lane >> 3) & 1)][8 * (lane >> 4)]);
  ldsm_x4(bf, &BE[(lane & 7) + 8 * (lane >> 4)][8 * ((lane >> 3) & 1)]);
  float d[4] = {0x1p-86f, 0x1p-86f, 0x1p-86f, 0x1p-86f};
  mma(d, af, bf);
  const int g = lane >> 2, t = lane & 3;
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p) out[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p] = __float_as_int(d[2 * h + p]);
}

__global__ void k_mma_raw(const uint16_t* A, const uint16_t* B, const float* C, int32_t* out) {
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
  ldsm_x4(af, &AE[(lane & 7) + 8 * ((lane >> 3) & 1)][8 * (lane >> 4)]);
  ldsm_x4(bf, &BE[(lane & 7) + 8 * (lane >> 4)][8 * ((lane >> 3) & 1)]);
  const int g = lane >> 2, t = lane & 3;
  float d[4];
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p) d[2 * h + p] = C[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p];
  mma(d, af, bf);
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 2; ++p) out[(size_t)tile * 128 + (g + 8 * h) * 8 + 2 * t + p] = __float_as_int(d[2 * h + p]);
}

torch::Tensor epass_raw(torch::Tensor a, torch::Tensor b) {  // a [T*16,16] uint8, b [T*8,16] uint8 (E4M3 bytes)
  const int T = a.size(0) / 16;
  auto out = torch::empty({T, 16, 8}, a.options().dtype(torch::kInt32));
  k_epass_raw<<<T, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), out.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
torch::Tensor mma_bf16_raw(torch::Tensor a, torch::Tensor b, torch::Tensor c) {  // a [T*16,16] int16, b [T*8,16] int16, c [T,16,8] f32
  const int T = a.size(0) / 16;
  auto out = torch::empty({T, 16, 8}, a.options().dtype(torch::kInt32));
  k_mma_raw<<<T, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint16_t*>(a.data_ptr<int16_t>()),
      reinterpret_cast<const uint16_t*>(b.data_ptr<int16_t>()), c.data_ptr<float>(), out.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("epass_raw", &epass_raw); m.def("mma_bf16_raw", &mma_bf16_raw); }
