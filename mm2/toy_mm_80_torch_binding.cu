#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

extern "C" __global__ void tc_mma_cpasync_inline(
    const half* A,
    const half* B,
    const float* C,
    float* D);

namespace {

void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(b.scalar_type() == torch::kFloat16, "B must be float16");
  TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(a.numel() >= 256, "A must have at least 256 half values");
  TORCH_CHECK(b.numel() >= 256, "B must have at least 256 half values");
}

}  // namespace

torch::Tensor toy_mm80(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);

  c10::cuda::CUDAGuard device_guard(a.device());
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");

  auto c = torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
  auto d = torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

  constexpr int threads = 32;
  constexpr int blocks = 1;
  constexpr int smem_bytes = 4096;

  auto stream = at::cuda::getDefaultCUDAStream();
  tc_mma_cpasync_inline<<<blocks, threads, smem_bytes, stream>>>(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      c.data_ptr<float>(),
      d.data_ptr<float>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("toy_mm80", &toy_mm80, "Launch toy sm80 mma kernel (CUDA)");
}
