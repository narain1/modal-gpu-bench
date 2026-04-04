// sm80_inline_ptx_mma.cu
// nvcc -O3 -arch=sm_80 sm80_inline_ptx_mma.cu -o a.out

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_e));                                          \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t id;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* smem_ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void cp_async_ca_16B(uint32_t dst_smem_u32, const void* src_gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst_smem_u32), "l"(src_gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_group_0()  { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync_0()             { asm volatile("bar.sync 0;\n" ::: "memory"); }

__device__ __forceinline__ void ldmatrix_m8n8_x4(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3, uint32_t smem_u32) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(smem_u32));
}

__device__ __forceinline__ void ldmatrix_m8n8_x2_trans(uint32_t &b0, uint32_t &b1, uint32_t smem_u32) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b0), "=r"(b1) : "r"(smem_u32));
}

__device__ __forceinline__ void mma_m16n8k16_rowcol_f32f16f16f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
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

extern "C" __global__ void tc_mma_cpasync_inline(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D) {
  (void)C;
  if (threadIdx.x >= 32) return;

  extern __shared__ uint8_t smem[];
  // Row-major staging buffers populated via cp.async.
  half* smemA_row = reinterpret_cast<half*>(smem);             // 256 half = 512B
  half* smemB_row = reinterpret_cast<half*>(smem + 512);       // 256 half = 512B
  // Packed tile buffers for ldmatrix.
  half* smemA_pack = reinterpret_cast<half*>(smem + 1024);     // 4x(8x8) = 256 half
  half* smemB_pack = reinterpret_cast<half*>(smem + 1536);     // 2x(8x8) = 128 half

  const uint32_t lane = lane_id();
  const uint32_t byte_off = lane * 16;

  // Copy 512B of A and 512B of B from global to shared via cp.async.
  cp_async_ca_16B(cvta_to_shared_u32(reinterpret_cast<uint8_t*>(smemA_row) + byte_off), (const uint8_t*)A + byte_off);
  cp_async_ca_16B(cvta_to_shared_u32(reinterpret_cast<uint8_t*>(smemB_row) + byte_off), (const uint8_t*)B + byte_off);
  cp_async_commit_group();
  cp_async_wait_group_0();
  bar_sync_0();

  // Repack A from row-major 16x16 into four contiguous 8x8 tiles.
  for (int idx = static_cast<int>(lane); idx < 256; idx += 32) {
    const int r = idx / 16;
    const int c = idx % 16;
    const int tile = (r / 8) * 2 + (c / 8);
    const int rr = r % 8;
    const int cc = c % 8;
    smemA_pack[tile * 64 + rr * 8 + cc] = smemA_row[idx];
  }

  // Repack B from row-major 16x8 into two contiguous 8x8 tiles.
  for (int idx = static_cast<int>(lane); idx < 128; idx += 32) {
    const int r = idx / 8;
    const int c = idx % 8;
    const int tile = r / 8;
    const int rr = r % 8;
    smemB_pack[tile * 64 + rr * 8 + c] = smemB_row[idx];
  }
  bar_sync_0();

  // Lane-wise row addresses for ldmatrix.
  // A uses x4: four 8x8 row-major tiles from a 16x16 source.
  const uint32_t lane_row = lane & 7;
  const uint32_t lane_group_a = lane >> 3;          // 0..3
  const uint32_t a_addr = cvta_to_shared_u32(
      reinterpret_cast<uint8_t*>(smemA_pack) + lane_group_a * 128 + lane_row * 16);

  // B uses x2.trans: two 8x8 tiles (KxN = 16x8), transposed on load.
  const uint32_t lane_group_b = (lane >> 3) & 1;    // 0..1 repeating across warp
  const uint32_t b_addr = cvta_to_shared_u32(
      reinterpret_cast<uint8_t*>(smemB_pack) + lane_group_b * 128 + lane_row * 16);

  uint32_t a0, a1, a2, a3;
  uint32_t b0, b1;
  ldmatrix_m8n8_x4(a0, a1, a2, a3, a_addr);
  ldmatrix_m8n8_x2_trans(b0, b1, b_addr);

  float d0, d1, d2, d3;
  mma_m16n8k16_rowcol_f32f16f16f32(
      d0, d1, d2, d3,
      a0, a1, a2, a3,
      b0, b1,
      0.0f, 0.0f, 0.0f, 0.0f);

  if (lane == 0) {
    D[0] = d0;
    D[1] = d1;
    D[2] = d2;
    D[3] = d3;
  }
}

// // -----------------------
// // main()
// // -----------------------
// int main() {
//   CUDA_CHECK(cudaSetDevice(0));

//   // The toy cp.async pattern copies 32 lanes * 16B = 512 bytes from A and B each.
//   // Each 16B chunk is 8 half elements, so total half elements needed: 512 / 2 = 256.
//   constexpr int kHalfElems = 256;
//   constexpr size_t kBytesAB = kHalfElems * sizeof(half);

//   std::vector<half> hA(kHalfElems), hB(kHalfElems);
//   for (int i = 0; i < kHalfElems; ++i) {
//     // simple, deterministic values
//     hA[i] = __float2half((i % 13) * 0.25f);
//     hB[i] = __float2half((i % 7)  * 0.5f);
//   }

//   // C is unused in this demo kernel (accumulators are set to 0), but allocate anyway.
//   std::vector<float> hC(4, 0.0f);
//   std::vector<float> hD(4, -1.0f);

//   half*  dA = nullptr;
//   half*  dB = nullptr;
//   float* dC = nullptr;
//   float* dD = nullptr;

//   CUDA_CHECK(cudaMalloc(&dA, kBytesAB));
//   CUDA_CHECK(cudaMalloc(&dB, kBytesAB));
//   CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
//   CUDA_CHECK(cudaMalloc(&dD, hD.size() * sizeof(float)));

//   CUDA_CHECK(cudaMemcpy(dA, hA.data(), kBytesAB, cudaMemcpyHostToDevice));
//   CUDA_CHECK(cudaMemcpy(dB, hB.data(), kBytesAB, cudaMemcpyHostToDevice));
//   CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));
//   CUDA_CHECK(cudaMemset(dD, 0, hD.size() * sizeof(float)));

//   // Shared memory: match the PTX toy (original had 4096). Over-alloc is fine for this demo.
//   constexpr size_t kSmemBytes = 4096;

//   dim3 grid(1);
//   dim3 block(32);

//   tc_mma_cpasync_inline<<<grid, block, kSmemBytes>>>(dA, dB, dC, dD);
//   CUDA_CHECK(cudaGetLastError());
//   CUDA_CHECK(cudaDeviceSynchronize());

//   CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(float), cudaMemcpyDeviceToHost));

//   printf("D[0..3] = {%f, %f, %f, %f}\n", hD[0], hD[1], hD[2], hD[3]);

//   CUDA_CHECK(cudaFree(dA));
//   CUDA_CHECK(cudaFree(dB));
//   CUDA_CHECK(cudaFree(dC));
//   CUDA_CHECK(cudaFree(dD));

//   return 0;
// }
