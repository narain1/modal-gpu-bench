import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
// Implements a Tensor Core GEMM using:
//   - PTX mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   - PTX cp.async (global->shared) with an N-stage pipeline
//   - PTX ldmatrix (shared->register fragments)
//
//   - M, N, K multiples of 128, 128, 32 respectively (specialized fast path)
// Notes:
//   - Uses shared-memory swizzle (permuted layout) to reduce bank conflicts
//   - The code is intentionally specialized and not a full CUTLASS replacement

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#error "This kernel requires sm80+ (cp.async, ldmatrix, mma)."
#endif

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
  // Copies 16 bytes. Uses .cg (cache global) and an L2 hint.
  // smem address must be in 32-bit shared address space.
  unsigned smem_u32 = __cvta_generic_to_shared(smem_dst);
  asm volatile(
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(smem_u32), "l"(gmem_src), "n"(16));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* smem_addr) {
  unsigned smem_u32 = __cvta_generic_to_shared(smem_addr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_u32));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, const void* smem_addr) {
  unsigned smem_u32 = __cvta_generic_to_shared(smem_addr);
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_u32));
}

__device__ __forceinline__ void mma_m16n8k16_f32f16f16(
    float& d0, float& d1, float& d2, float& d3,
    const unsigned& a0, const unsigned& a1, const unsigned& a2, const unsigned& a3,
    const unsigned& b0, const unsigned& b1,
    const float& c0, const float& c1, const float& c2, const float& c3) {

  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, "
      "{%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// ----------------------------- Kernel -----------------------------
//
// Tile choices (like the referenced blog's "3.1" spirit):
//   - Threadblock output tile: 128 x 128 (fp32)
//   - K step per iteration: 32 (two mma k=16 steps via ldmatrix XOR trick)
//   - Warps per block: 8 (256 threads)
//   - Each warp computes: 64 x 64 /? We use 4x4 warp-tiles of 16x16 -> 64x64 per warp-group,
//     and distribute across 8 warps so that TB computes 128x128.
//
// This is a specialized kernel for sizes multiple of 128 and K multiple of 32.

template<int N_STAGES>
__global__ void tc_gemm_async_mma_128x128x32(
    const half* __restrict__ A,    // [M,K] row-major
    const half* __restrict__ Bcol, // [N,K] row-major == column-major of original B (KxN)
    float* __restrict__ C,         // [M,N] row-major
    int M, int N, int K) {

  // 256 threads, 8 warps
  const int tid  = threadIdx.x;
  const int warp = tid >> 5;     // 0..7
  const int lane = tid & 31;     // 0..31

  // TB tile origin in C
  const int tb_m0 = blockIdx.y * 128;
  const int tb_n0 = blockIdx.x * 128;

  // Shared buffers: store uint4 vectors (16B) == 8 fp16
  // We store A/B tiles in K-major vectors (K/8 columns).
  // For each K-step=32, we need 32/8=4 uint4 columns.
  //
  // We stage a (128 x 32) A tile and a (32 x 128) B tile per iteration.
  // In uint4 terms:
  //   A: 128 rows, 32 cols -> 4 uint4 per row  => 128*4 = 512 uint4
  //   B: 128 cols (as N-dim), 32 rows (K-dim). For row.col mma, we treat B as col-major KxN,
  //      and we load it as vectors along K (contiguous in Bcol because Bcol is [N,K] row-major).
  //      That is: for each output-column n, we load 32 K values (4 uint4).
  //      => 128 * 4 = 512 uint4
  //
  // We use a swizzled (permuted) [row][col] layout for bank-conflict reduction.
  // Layout: [N_STAGES][128][4] of uint4 for A and B.
  __shared__ uint4 As[N_STAGES][128][4];
  __shared__ uint4 Bs[N_STAGES][128][4];

  // Swizzle for storing/loading uint4 in shmem (bank-conflict mitigation)
  // Similar idea to XORing col by row-group.
  auto swizzle_col = [](int row, int col) {
    // row in [0,127], col in [0,3]
    // Spread col across banks by mixing low row bits.
    return col ^ ((row >> 3) & 0x3);
  };

  // Thread mapping for cp.async loads:
  //  - First 256 threads cooperatively load 1024 uint4 per stage (512 A + 512 B).
  //  - Each thread loads 4 uint4 total per stage (if tid < 256, indices 0..1023).
  constexpr int VEC_A = 512;
  constexpr int VEC_B = 512;
  constexpr int VEC_T = VEC_A + VEC_B; // 1024

  // Global pointers to tile starts
  const half* A_tile = A + tb_m0 * K;      // A[tb_m0, 0]
  const half* B_tile = Bcol + tb_n0 * K;   // Bcol[tb_n0, 0]  (remember Bcol is [N,K])

  // Utility to launch cp.async for one stage at k0 (in units of 8 fp16 elements)
  auto stage_g2s = [&](int stage, int k0_vec) {
    // k0_vec is in [0, K/8)
    // Load A: for each row m in 0..127, load 4 uint4 spanning 32 K values (k0_vec..k0_vec+3)
    // A row-major contiguous along K.
    for (int i = tid; i < VEC_T; i += blockDim.x) {
      if (i < VEC_A) {
        int m = i >> 2;      // 0..127
        int c = i & 3;       // 0..3
        int sc = swizzle_col(m, c);
        const uint4* gsrc = reinterpret_cast<const uint4*>(A_tile + m * K + (k0_vec + c) * 8);
        cp_async_16B(&As[stage][m][sc], gsrc);
      } else {
        int j = i - VEC_A;
        int n = j >> 2;      // 0..127
        int c = j & 3;       // 0..3
        int sc = swizzle_col(n, c);
        // Bcol is [N,K] row-major, contiguous along K.
        const uint4* gsrc = reinterpret_cast<const uint4*>(B_tile + n * K + (k0_vec + c) * 8);
        cp_async_16B(&Bs[stage][n][sc], gsrc);
      }
    }
    cp_async_commit();
  };

  // Preload N_STAGES-1
  const int Kvec = K >> 3;           // K/8
  const int Kstep_vec = 4;           // 32/8
  int preload = N_STAGES - 1;
  #pragma unroll
  for (int s = 0; s < N_STAGES - 1; ++s) {
    int k0 = s * Kstep_vec;
    stage_g2s(s, k0);
  }

  // Accumulators for a 64x64 per warp arrangement:
  // We map warps into a 2x4 grid over the 128x128 tile,
  // each warp computes a 64x32 region via 4x2 tiles of 16x16.
  // For simplicity, each warp computes 4 (m-tiles) x 2 (n-tiles) = 8 tiles of 16x16.
  // Each 16x16 is two mma (n=8 twice). Each mma returns 4 floats per thread.
  //
  // d[mTile][nTile][2 halves] each holds float4 per thread.
  float d[4][2][2][4];
  #pragma unroll
  for (int mi = 0; mi < 4; ++mi)
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni)
      #pragma unroll
      for (int hi = 0; hi < 2; ++hi)
        #pragma unroll
        for (int t = 0; t < 4; ++t)
          d[mi][ni][hi][t] = 0.f;

  // Warp's base within the 128x128 output tile:
  // warp row group: 0..1, warp col group: 0..3
  int warp_mg = warp >> 2;        // 0..1
  int warp_ng = warp & 3;         // 0..3
  int warp_m0 = warp_mg * 64;     // 0 or 64
  int warp_n0 = warp_ng * 32;     // 0,32,64,96

  // ldmatrix addressing within swizzled [row][col] of uint4:
  // Each uint4 is 8 fp16 along K. ldmatrix loads 8x8 b16 tiles from shared.
  // We follow the same trick as the referenced post: load k=0..1 and k=2..3 via xor on col.
  int loadRowA = (lane % 16) / 2;
  int loadRowB = (lane % 8) / 2;

  // The col index we present to ldmatrix is in units of uint4 slots (each is 8 fp16).
  // We need a mapping that matches the swizzle.
  auto shmem_ptr_A = [&](int stage, int m_row, int c) -> const void* {
    int sc = swizzle_col(m_row, c);
    return (const void*)(&As[stage][m_row][sc]);
  };
  auto shmem_ptr_B = [&](int stage, int n_col, int c) -> const void* {
    int sc = swizzle_col(n_col, c);
    return (const void*)(&Bs[stage][n_col][sc]);
  };

  // Main loop over K in 32-wide steps
  int stages_in_flight = N_STAGES - 1;
  int num_iters = (Kvec / Kstep_vec);

  for (int it = 0; it < num_iters; ++it) {
    int load_stage  = it % N_STAGES;
    int store_stage = (it + N_STAGES - 1) % N_STAGES;

    // Wait so that <= N_STAGES-2 groups remain pending
    cp_async_wait_group<N_STAGES - 2>();
    __syncthreads();

    // Launch next async copy (keep wait arg constant; clamp near end)
    int next_k0 = (it + N_STAGES - 1) * Kstep_vec;
    if (next_k0 >= Kvec) next_k0 = Kvec - Kstep_vec;
    stage_g2s(store_stage, next_k0);

    // Compute on the loaded stage:
    // For each warp: 4 m-tiles (16 each => 64) and 2 n-tiles (16 each => 32)
    // Each 16x16 uses:
    //   - two mma for n halves (n=0..7 and n=8..15)
    //   - and two k-halves (k=0..15 and k=16..31) achieved by loading c=0..1 and c=2..3
    //
    // Registers for fragments:
    unsigned a0,a1,a2,a3;
    unsigned b0,b1;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
      int m_tile_row = warp_m0 + mi * 16;
      int m_row = m_tile_row + loadRowA;  // row inside 128 tile

      // Two A loads: c=0..1 (we'll use ldmatrix x4, but our shared stores uint4 by c index)
      // Here we present the address for the first 8x128b matrix row; ldmatrix fetches the rest warp-wide.
      // We load A for k0..15 using c=0 and for k16..31 using c=2 (xor 2).
      // This relies on the PTX fragment layout expectations (as in the blog).
      //
      // Load A (k0..15):
      ldmatrix_x4(a0,a1,a2,a3, shmem_ptr_A(load_stage, m_row, 0));
      unsigned a0b,a1b,a2b,a3b;
      // Load A (k16..31):
      ldmatrix_x4(a0b,a1b,a2b,a3b, shmem_ptr_A(load_stage, m_row, 2));

      #pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        int n_tile_col = warp_n0 + ni * 16;
        // Map lanes 0-31 to columns 0-15 within the 16-wide tile
        // Each lane handles one element in the 8-wide mma output
        int lane_in_tile = lane & 15;  // 0-15
        int n_col = n_tile_col + lane_in_tile;

        // Load B (k0..15): ldmatrix x2
        ldmatrix_x2(b0,b1, shmem_ptr_B(load_stage, n_col, 0));
        unsigned b0b,b1b;
        // Load B (k16..31):
        ldmatrix_x2(b0b,b1b, shmem_ptr_B(load_stage, n_col, 2));

        // Each mma gives 16x8, so do two halves in N for 16x16:
        // First MMA for columns 0-7, second MMA for columns 8-15

        // N-half 0 (columns 0-7)
        {
          float &d0 = d[mi][ni][0][0], &d1 = d[mi][ni][0][1], &d2 = d[mi][ni][0][2], &d3 = d[mi][ni][0][3];
          mma_m16n8k16_f32f16f16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, d0,d1,d2,d3);
          mma_m16n8k16_f32f16f16(d0,d1,d2,d3, a0b,a1b,a2b,a3b, b0b,b1b, d0,d1,d2,d3);
        }

        // N-half 1 (columns 8-15) - need to load different B columns (n_col + 8)
        {
          int n_col_hi = n_col + 8;
          ldmatrix_x2(b0,b1, shmem_ptr_B(load_stage, n_col_hi, 0));
          ldmatrix_x2(b0b,b1b, shmem_ptr_B(load_stage, n_col_hi, 2));
          float &d0 = d[mi][ni][1][0], &d1 = d[mi][ni][1][1], &d2 = d[mi][ni][1][2], &d3 = d[mi][ni][1][3];
          mma_m16n8k16_f32f16f16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, d0,d1,d2,d3);
          mma_m16n8k16_f32f16f16(d0,d1,d2,d3, a0b,a1b,a2b,a3b, b0b,b1b, d0,d1,d2,d3);
        }
      }
    }

    __syncthreads();
  }

  // Epilogue: store fp32 accumulators to C.
  // Direct per-lane storage: each lane stores its 4-element vector to one output row
  int base_m = tb_m0 + warp_m0;
  int base_n = tb_n0 + warp_n0;

  // Map lane to (mi, ni, t):
  // - 32 lanes -> 8 lanes per mi (4 mi values = 32)
  // - Within each mi group: 4 lanes per ni (2 ni values = 8, close enough)
  // - t = lane % 4
  int mi = lane >> 3;    // 0-3 (8 lanes each)
  int ni = (lane >> 2) & 1; // 0-1 (4 lanes each)
  int t = lane & 3;      // 0-3 (element within the mma result)

  // Each thread stores to one row of the output
  // Row = base_m + mi*16 + (lane % 8) -- spread across the 16 rows of the m-tile
  int row_within_tile = (lane % 8);
  int m = base_m + mi * 16 + row_within_tile;
  int n_base = base_n + ni * 16;

  // Store N-half 0 (columns 0-7)
  if (m < M) {
    C[m * N + n_base + t] = d[mi][ni][0][t];
    // Store N-half 1 (columns 8-15)
    if (n_base + 8 + t < N) {
      C[m * N + n_base + 8 + t] = d[mi][ni][1][t];
    }
  }
}

// ----------------------------- C++ binding -----------------------------

torch::Tensor mm_tc_async(torch::Tensor A, torch::Tensor Bcol) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(Bcol.is_cuda(), "Bcol must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(Bcol.dtype() == torch::kFloat16, "Bcol must be fp16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(Bcol.is_contiguous(), "Bcol must be contiguous (row-major), typically B.t().contiguous()");
  TORCH_CHECK(A.dim() == 2 && Bcol.dim() == 2, "A and Bcol must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(Bcol.size(1) == K, "Bcol must have shape [N,K] with same K as A");
  int64_t N = Bcol.size(0);

  // Specialized constraints
  TORCH_CHECK((M % 128) == 0, "M must be multiple of 128");
  TORCH_CHECK((N % 128) == 0, "N must be multiple of 128");
  TORCH_CHECK((K % 32)  == 0, "K must be multiple of 32");

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

  constexpr int THREADS = 256;
  constexpr int NSTAGES = 3;

  dim3 block(THREADS);
  dim3 grid((unsigned)(N / 128), (unsigned)(M / 128));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  tc_gemm_async_mma_128x128x32<NSTAGES><<<grid, block, 0, stream>>>(
      (half*)A.data_ptr<at::Half>(),
      (half*)Bcol.data_ptr<at::Half>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

TORCH_LIBRARY(lib, m) {
  m.def("mm(Tensor A, Tensor Bcol) -> Tensor");
  m.impl("mm", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(mm_tc_async)));
}
"""

mod = load_inline(
    name="tma_mm_demo",
    cpp_sources="",
    cuda_sources=SRC,
    functions=None,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90"],
    with_cuda=True,
    verbose=True,
    is_python_module=False,
)

# exposed as torch.ops.lib.mm

# Benchmark: 4096x4096 matrix multiply with warmup and timing
import time

M, K, N = 4096, 4096, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
Bcol = B.t().contiguous()
C = torch.ops.lib.mm(A, Bcol)

torch.testing.assert_close(C.to(torch.float16), torch.matmul(A, B.t()))

# Warmup runs
print("Warming up...")
for _ in range(10):
    C = torch.ops.lib.mm(A, Bcol)

# Synchronize before timing
torch.cuda.synchronize()

# Timed runs
num_runs = 100
start = time.perf_counter()
for _ in range(num_runs):
    C = torch.ops.lib.mm(A, Bcol)
torch.cuda.synchronize()
end = time.perf_counter()

avg_ms = (end - start) / num_runs * 1000
print(f"Average time per matmul (4096x4096): {avg_ms:.6f} ms")

# Calculate TFLOPS (2*M*K*N floating point ops)
tflops = 2 * M * N * K / (avg_ms * 1e-3) / 1e12
print(f"TFLOPS: {tflops:.6f}")

# Bandwidth calculation (bytes read + written)
bytes_transferred = (M * K + K * N + M * N) * 2  # fp16 = 2 bytes
bandwidth_gbps = bytes_transferred / (avg_ms * 1e-3) / 1e9
print(f"Bandwidth: {bandwidth_gbps:.6f} GB/s")