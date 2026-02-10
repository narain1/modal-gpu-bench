#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(status)                                                                 \
  {                                                                                        \
    cudaError_t error = (status);                                                          \
    if (error != cudaSuccess) {                                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return -1;                                                                           \
    }                                                                                      \
  }

#define CUTLASS_CHECK(status)                                                              \
  {                                                                                        \
    cutlass::Status result = (status);                                                     \
    if (result != cutlass::Status::kSuccess) {                                             \
      std::cerr << "CUTLASS error at " << __FILE__ << ":" << __LINE__                   \
                << " status = " << static_cast<int>(result) << std::endl;                \
      return -1;                                                                           \
    }                                                                                      \
  }

int main() {
  using ElementA = float;
  using ElementB = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementC = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  // SIMT requires alignment of 1
  constexpr int AlignmentA = 1;
  constexpr int AlignmentB = 1;
  constexpr int AlignmentC = 1;

  using OperatorClass = cutlass::arch::OpClassSimt;
  using ArchTag = cutlass::arch::Sm80;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  constexpr int NumStages = 2;
  constexpr int EVTEpilogueStages = 1;

  using namespace cutlass::epilogue::threadblock;
  using namespace cute;

  using OutputTileThreadMap = OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementC,
      AlignmentC,
      EVTEpilogueStages>;

  using AccumulatorFetch = VisitorAccFetch;
  using BiasBroadcast = VisitorRowBroadcast<
      OutputTileThreadMap,
      ElementCompute,
      Stride<_0, _1, int32_t>>;

  using AddBias = VisitorCompute<
      cutlass::plus,
      ElementCompute,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using BiasComputeTree = Sm80EVT<AddBias, AccumulatorFetch, BiasBroadcast>;

  using OutputStore = VisitorAuxStore<
      OutputTileThreadMap,
      ElementC,
      cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, _1, int64_t>>;

  using EpilogueTree = Sm80EVT<OutputStore, BiasComputeTree>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      AlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      AlignmentB,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementAccumulator,
      ElementCompute,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueTree,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      EVTEpilogueStages>::GemmKernel;

  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
  if (props.major * 10 + props.minor < 80) {
    std::cerr << "Requires SM80+ GPU for EVT epilogue demo." << std::endl;
    return 0;
  }

  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 128;

  std::vector<ElementA> host_A(M * K);
  std::vector<ElementB> host_B(K * N);
  std::vector<ElementC> host_D(M * N, ElementC(0));
  std::vector<ElementCompute> host_bias(N);
  std::vector<ElementC> reference(M * N, ElementC(0));

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &value : host_A) {
    value = static_cast<ElementA>(dist(rng));
  }
  for (auto &value : host_B) {
    value = static_cast<ElementB>(dist(rng));
  }
  for (auto &value : host_bias) {
    value = static_cast<ElementCompute>(dist(rng));
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      ElementCompute acc = host_bias[n];
      for (int k = 0; k < K; ++k) {
        acc += ElementCompute(host_A[m * K + k]) * ElementCompute(host_B[k * N + n]);
      }
      reference[m * N + n] = static_cast<ElementC>(acc);
    }
  }

  cutlass::device_memory::allocation<ElementA> device_A(M * K);
  cutlass::device_memory::allocation<ElementB> device_B(K * N);
  cutlass::device_memory::allocation<ElementC> device_D(M * N);
  cutlass::device_memory::allocation<ElementCompute> device_bias(N);

  CUDA_CHECK(cudaMemcpy(device_A.get(), host_A.data(), sizeof(ElementA) * host_A.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_B.get(), host_B.data(), sizeof(ElementB) * host_B.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_bias.get(), host_bias.data(), sizeof(ElementCompute) * host_bias.size(), cudaMemcpyHostToDevice));

  // Compose visitor callbacks: add the bias vector to the accumulator tile before storing.
  auto output_stride = cute::make_stride(int64_t(N), _1{}, int64_t(M) * int64_t(N));
  auto bias_stride = cute::make_stride(_0{}, _1{}, int32_t(N));

  typename EpilogueTree::Arguments callback_args{
      {device_D.get(), output_stride},  // OutputStore args
      {
          {},  // AddBias args
          {    // TreeVisitor2x<AccumulatorFetch, BiasBroadcast> args
              {},  // AccumulatorFetch args
              {device_bias.get(), ElementCompute(0), bias_stride}  // BiasBroadcast args
          }
      }
  };

  typename DeviceGemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      1,
      callback_args,
      device_A.get(),
      device_B.get(),
      nullptr,
      nullptr,
      int64_t(M) * K,
      int64_t(K) * N,
      0,
      0,
      K,
      N,
      0,
      0);

  DeviceGemm gemm_op;
  size_t workspace_size = DeviceGemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op());

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_D.data(), device_D.get(), sizeof(ElementC) * host_D.size(), cudaMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (size_t idx = 0; idx < host_D.size(); ++idx) {
    max_diff = std::max(max_diff, std::abs(host_D[idx] - reference[idx]));
  }

  std::cout << "Max difference vs. host reference: " << max_diff << std::endl;
  std::cout << "Sample output tile (0, 0..3):";
  for (int n = 0; n < 4; ++n) {
    std::cout << " " << host_D[n];
  }
  std::cout << std::endl;

  return 0;
}
