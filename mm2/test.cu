#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <stdint.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <random>

#define M_TILE 2
#define N_TILE 2

__forceinline__ 
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B, float *C, float *D) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
     );
}


// Kernel 1.0: Naive mma 
__launch_bounds__(16 * 16)
__global__ void mma_matmul_1_0(const half *A, const half *B, float *C, int M, int N, int K) {
  // declare cache in shared memory
  __shared__ half As[32][16];
  __shared__ half Bs[16][32];

  int mBlock = 32 * blockIdx.y;
  int nBlock = 32 * blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // warps arranged in 2x4 grid:
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;

  // warp offsets in threadblock shmem tiles
  int nWarp = 8 * (warpID % 4);
  int mWarp = 16 * (warpID / 4);

  // warps are split into 8 groups of 4 threads each
  int groupID     = laneID / 4;
  int groupLaneID = laneID % 4;

  half  aReg[8];
  half  bReg[4]; 
  float dReg[4] = {0.};

  for (int kStart=0; kStart < K; kStart += 16) {
    As[ty     ][tx] = A[(mBlock + ty     )*K + kStart + tx];
    As[ty + 16][tx] = A[(mBlock + ty + 16)*K + kStart + tx];
    Bs[ty][tx     ] = B[(kStart + ty)*K + nBlock      + tx];
    Bs[ty][tx + 16] = B[(kStart + ty)*K + nBlock + 16 + tx];
    __syncthreads();

    // set up the registers for mma call
    aReg[0] = As[mWarp + groupID    ][groupLaneID*2    ];
    aReg[1] = As[mWarp + groupID    ][groupLaneID*2 + 1];
    aReg[2] = As[mWarp + groupID + 8][groupLaneID*2    ];
    aReg[3] = As[mWarp + groupID + 8][groupLaneID*2 + 1];
    aReg[4] = As[mWarp + groupID    ][groupLaneID*2 + 8];
    aReg[5] = As[mWarp + groupID    ][groupLaneID*2 + 9];
    aReg[6] = As[mWarp + groupID + 8][groupLaneID*2 + 8];
    aReg[7] = As[mWarp + groupID + 8][groupLaneID*2 + 9];

    bReg[0] = Bs[groupLaneID*2 + 0][nWarp + groupID];
    bReg[1] = Bs[groupLaneID*2 + 1][nWarp + groupID];
    bReg[2] = Bs[groupLaneID*2 + 8][nWarp + groupID];
    bReg[3] = Bs[groupLaneID*2 + 9][nWarp + groupID];
    unsigned const *aPtr = reinterpret_cast<unsigned const *>(&aReg);
    unsigned const *bPtr = reinterpret_cast<unsigned const *>(&bReg);
    mma_m16n8k16(aPtr, bPtr, dReg, dReg);
    __syncthreads();
  }
  // Write results to global memory
  C[(mBlock + mWarp + groupID)*N + nBlock + nWarp + 2*groupLaneID] = dReg[0];
  C[(mBlock + mWarp + groupID)*N + nBlock + nWarp + 2*groupLaneID+1] = dReg[1];
  C[(mBlock + mWarp + groupID+8)*N + nBlock + nWarp + 2*groupLaneID] = dReg[2];
  C[(mBlock + mWarp + groupID+8)*N + nBlock + nWarp + 2*groupLaneID+1] = dReg[3];
}


bool validate(float* h_C, int M, int N, int K, half *a, half *b) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(a[i*K + k]) * __half2float(b[k*N + j]);
            }
            if (fabs(sum - h_C[i*N + j]) > 1e-2f) {
                printf("Mismatch at (%d, %d): expected %f, got %f\n", i, j, sum, h_C[i*N + j]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    float *h_C = new float[M * N];
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(rand() / (float)RAND_MAX);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(rand() / (float)RAND_MAX);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;
    
    half *d_A; cudaMalloc(&d_A, M * K * sizeof(half));
    half *d_B; cudaMalloc(&d_B, K * N * sizeof(half));
    float *d_C; cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(N / (N_TILE * blockDim.x), M / (M_TILE * blockDim.y));
    mma_matmul_1_0<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);    

    if (validate(h_C, M, N, K, h_A, h_B))
        printf("Validation PASSED\n");
    else
        printf("Validation FAILED\n");
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}