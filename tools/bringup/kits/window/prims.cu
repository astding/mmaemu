// Exposes the device primitives of v2_common.cuh to Python for exhaustive / adversarial testing (test_prims.py).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "v2_common.cuh"

namespace {
using namespace p2;

__global__ void k_decode(int32_t* out) {  // [256, 5]: v bits, o32, o16, la, sg
  const int b = threadIdx.x;
  const Dec d = decode(b);
  out[b * 5 + 0] = __float_as_int(d.v); out[b * 5 + 1] = d.o32; out[b * 5 + 2] = d.o16;
  out[b * 5 + 3] = d.la; out[b * 5 + 4] = d.sg;
}

// E-pass over one 16-term block per row, exactly as in the GEMM kernels: returns fp32 bits of 2^E_prod.
// mode 0: bf16x2 (lane 0 and lane 1 computed separately and both returned), mode 1: fp32.
__global__ void k_epass(const uint8_t* a, const uint8_t* b, int32_t* out, int n, int mode) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  if (mode == 0) {
    uint32_t acc = EPASS_INIT_BF16X2;
    for (int k = 0; k < 16; ++k) {
      const Dec da = decode(a[r * 16 + k]), db = decode(b[r * 16 + k]);
      // lane 0: this row's term; lane 1: the same term (A operand broadcast {o,o}, B pair {o_j, o_j'} = {o, o})
      acc = bfma2(da.o16 | (da.o16 << 16), db.o16 | (db.o16 << 16), acc);
    }
    out[2 * r] = pbits_bf16((acc >> 7) & 0xffu);
    out[2 * r + 1] = pbits_bf16(acc >> 23);
  } else {
    float acc = EPASS_INIT_F32;
    for (int k = 0; k < 16; ++k) {
      const Dec da = decode(a[r * 16 + k]), db = decode(b[r * 16 + k]);
      acc = __fmaf_rn(__uint_as_float(da.o32), __uint_as_float(db.o32), acc);
    }
    out[2 * r] = out[2 * r + 1] = pbits_f32(__float_as_uint(acc) >> 23);
  }
}

__global__ void k_rz(const float* a, const float* b, const float* s, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __fmaf_rz(a[i], b[i], s[i]);
}

__global__ void k_init(const float* V, const int32_t* pb, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = block_init(V[i], (uint32_t)pb[i]);
}

__global__ void k_finish(const float* S, const int32_t* w, float* out, int n) {  // w: [n, 16] = la[4] sa[4] lb[4] sb[4]
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t la[4], sa[4], lb[4], sb[4];
  for (int q = 0; q < 4; ++q) {
    la[q] = w[i * 16 + q]; sa[q] = w[i * 16 + 4 + q]; lb[q] = w[i * 16 + 8 + q]; sb[q] = w[i * 16 + 12 + q];
  }
  out[i] = block_finish(S[i], la, sa, lb, sb);
}

__global__ void k_trunc(const float* V, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = final_trunc(V[i]);
}

// v3 primitives
__global__ void k_decode_word(const uint32_t* w, int32_t* out, int n) {  // out [n, 7]: sg la x v0..v3 (bits)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t sg, la, x; float v[4];
  decode_word(w[i], sg, la, x, v);
  out[i * 7 + 0] = sg; out[i * 7 + 1] = la; out[i * 7 + 2] = x;
  for (int q = 0; q < 4; ++q) out[i * 7 + 3 + q] = __float_as_int(v[q]);
}
// base-64 E-pass with byte-placed operands, as in v3: row r has 16 A bytes and 16 B bytes (one k each);
// lane 0 uses the A operand built by epa<q> and a B pair whose low half is built by epb<q> (high half: same term).
__global__ void k_epass64(const uint8_t* a, const uint8_t* b, int32_t* out, int n) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  uint32_t acc = EP64_INIT;
  for (int k = 0; k < 16; k += 4) {
    uint32_t wa = 0, wb = 0;
    for (int q = 0; q < 4; ++q) { wa |= (uint32_t)a[r * 16 + k + q] << (8 * q); wb |= (uint32_t)b[r * 16 + k + q] << (8 * q); }
    uint32_t sg, la, xa, xb; float v[4];
    decode_word(wa, sg, la, xa, v); decode_word(wb, sg, la, xb, v);
    const uint32_t A4[4] = {epa<0>(xa), epa<1>(xa), epa<2>(xa), epa<3>(xa)};
    const uint32_t B4[4] = {epb<0>(xb) & 0xffffu, epb<1>(xb) & 0xffffu, epb<2>(xb) & 0xffffu, epb<3>(xb) & 0xffffu};
    for (int q = 0; q < 4; ++q) acc = bfma2(A4[q], B4[q] | (B4[q] << 16), acc);
  }
  out[2 * r] = pbits64_raw((acc >> 7) & 0xffu) & 0x7f800000u;
  out[2 * r + 1] = pbits64_raw(acc >> 23) & 0x7f800000u;
}
__global__ void k_init3(const float* V, const int32_t* pb, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // pb: clean bits of 2^E_prod; add mantissa garbage to exercise the float-max path
  if (i < n) out[i] = block_init3(V[i], (uint32_t)pb[i] | ((uint32_t)(i * 2654435761u) & 0x007fffffu));
}

inline dim3 G(int n) { return dim3((n + 255) / 256); }
}  // namespace

torch::Tensor decode_table() {
  auto out = torch::empty({256, 5}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  k_decode<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int32_t>());
  return out;
}
torch::Tensor epass(torch::Tensor a, torch::Tensor b, int64_t mode) {
  const int n = a.size(0);
  auto out = torch::empty({n, 2}, a.options().dtype(torch::kInt32));
  k_epass<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(),
                                                                  out.data_ptr<int32_t>(), n, (int)mode);
  return out;
}
torch::Tensor rz(torch::Tensor a, torch::Tensor b, torch::Tensor s) {
  const int n = a.numel();
  auto out = torch::empty_like(a);
  k_rz<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(), b.data_ptr<float>(), s.data_ptr<float>(),
                                                              out.data_ptr<float>(), n);
  return out;
}
torch::Tensor init(torch::Tensor V, torch::Tensor pb) {
  const int n = V.numel();
  auto out = torch::empty_like(V);
  k_init<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(V.data_ptr<float>(), pb.data_ptr<int32_t>(),
                                                                out.data_ptr<float>(), n);
  return out;
}
torch::Tensor finish(torch::Tensor S, torch::Tensor w) {
  const int n = S.numel();
  auto out = torch::empty_like(S);
  k_finish<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(S.data_ptr<float>(), w.data_ptr<int32_t>(),
                                                                  out.data_ptr<float>(), n);
  return out;
}
torch::Tensor trunc14(torch::Tensor V) {
  const int n = V.numel();
  auto out = torch::empty_like(V);
  k_trunc<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(V.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}

torch::Tensor decode_word_t(torch::Tensor w) {
  const int n = w.numel();
  auto out = torch::empty({n, 7}, w.options().dtype(torch::kInt32));
  k_decode_word<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint32_t*)w.data_ptr<int32_t>(), out.data_ptr<int32_t>(), n);
  return out;
}
torch::Tensor epass64(torch::Tensor a, torch::Tensor b) {
  const int n = a.size(0);
  auto out = torch::empty({n, 2}, a.options().dtype(torch::kInt32));
  k_epass64<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), out.data_ptr<int32_t>(), n);
  return out;
}
torch::Tensor init3(torch::Tensor V, torch::Tensor pb) {
  const int n = V.numel();
  auto out = torch::empty_like(V);
  k_init3<<<G(n), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(V.data_ptr<float>(), pb.data_ptr<int32_t>(), out.data_ptr<float>(), n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_word", &decode_word_t); m.def("epass64", &epass64); m.def("init3", &init3);
  m.def("decode_table", &decode_table); m.def("epass", &epass); m.def("rz", &rz); m.def("init", &init);
  m.def("finish", &finish); m.def("trunc14", &trunc14);
}
