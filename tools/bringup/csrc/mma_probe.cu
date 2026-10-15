// Instruction-level probe: one mma.sync.aligned.m16n8k32.row.col.f32.{e4m3,e5m2}.{e4m3,e5m2}.f32 per test vector,
// with the operand layout of the MATLAB Tensor Core toolbox's validation kernel (model_validation/CUDA), so that the
// toolbox's published vectors can be measured on any FP8-capable GPU (sm_89+). Element (0, 0) of D is
//   d = sum_{k<32} a[k] * b[k] + c,
// a[k] placed in row 0 of A, b[k] in column 0 of B (PTX fragment layout for m16n8k32 with 8-bit operands).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

namespace {

template <bool A_E5M2, bool B_E5M2>
__global__ void mma_k32_kernel(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b,
                               const float* __restrict__ c, float* __restrict__ d, int n) {
  const int v = blockIdx.x;  // one warp per vector
  if (v >= n) return;
  const int lane = threadIdx.x;
  // A fragment (row-major 16x32): thread t holds rows {t/4, t/4+8}, k = (t%4)*4 + [0..3] and +16.
  // Row 0 is held by lanes 0..3: a0 (k = 4*lane .. +3), a2 (k = 16 + 4*lane .. +3); a1, a3 are row 8 (zero).
  uint32_t ra[4] = {0, 0, 0, 0};
  // B fragment (col-major 32x8): thread t holds column t/4, k = (t%4)*4 + [0..3] (b0) and +16 (b1).
  uint32_t rb[2] = {0, 0};
  float rc[4] = {0.f, 0.f, 0.f, 0.f};
  if (lane < 4) {
    const uint8_t* av = a + (size_t)v * 32;
    const uint8_t* bv = b + (size_t)v * 32;
    for (int q = 0; q < 4; ++q) {
      ra[0] |= (uint32_t)av[4 * lane + q] << (8 * q);
      ra[2] |= (uint32_t)av[16 + 4 * lane + q] << (8 * q);
      rb[0] |= (uint32_t)bv[4 * lane + q] << (8 * q);
      rb[1] |= (uint32_t)bv[16 + 4 * lane + q] << (8 * q);
    }
  }
  if (lane == 0) rc[0] = c[v];
  float rd[4];
#define MMA(TA, TB)                                                                                              \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32." TA "." TB ".f32 "                                       \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                    \
               : "=f"(rd[0]), "=f"(rd[1]), "=f"(rd[2]), "=f"(rd[3])                                              \
               : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]),                         \
                 "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3]))
  if (A_E5M2 && B_E5M2) MMA("e5m2", "e5m2");
  else if (A_E5M2) MMA("e5m2", "e4m3");
  else if (B_E5M2) MMA("e4m3", "e5m2");
  else MMA("e4m3", "e4m3");
#undef MMA
  if (lane == 0) d[v] = rd[0];  // D(0, 0)
}

}  // namespace

torch::Tensor mma_k32(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool a_e5m2, bool b_e5m2) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kUInt8 && a.dim() == 2 && a.size(1) == 32);
  TORCH_CHECK(b.sizes() == a.sizes() && b.scalar_type() == torch::kUInt8 && c.scalar_type() == torch::kFloat32);
  a = a.contiguous(); b = b.contiguous(); c = c.contiguous();
  const int n = a.size(0);
  auto d = torch::empty({n}, c.options());
  auto s = c10::cuda::getCurrentCUDAStream();
  auto* pa = a.data_ptr<uint8_t>(); auto* pb = b.data_ptr<uint8_t>(); auto* pc = c.data_ptr<float>(); auto* pd = d.data_ptr<float>();
  if (a_e5m2 && b_e5m2) mma_k32_kernel<true, true><<<n, 32, 0, s>>>(pa, pb, pc, pd, n);
  else if (a_e5m2) mma_k32_kernel<true, false><<<n, 32, 0, s>>>(pa, pb, pc, pd, n);
  else if (b_e5m2) mma_k32_kernel<false, true><<<n, 32, 0, s>>>(pa, pb, pc, pd, n);
  else mma_k32_kernel<false, false><<<n, 32, 0, s>>>(pa, pb, pc, pd, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mma_k32", &mma_k32, "one m16n8k32 FP8 MMA per vector, D(0,0)"); }
