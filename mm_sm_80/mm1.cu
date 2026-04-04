#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

__launch_bounds__(16 * 16)
__global__ void mm(const half *A, const half *B, half float *c, int M, int N, int K) {
  __shared__ half As[32][16];
  __shared__ half Bs[16][32];

  // block offset
  int mBlock = 32 * blockIdx.y;
  int nBlock = 16 * blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // linear threadId
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int wid = tid / 32; // warp id
  int lid = tid % 32; // lane id
  
  // warp offsets in threadblock shared memory
  int nWarp = 8 * (wid % 4); 
  int mWarp = 16 * (wid / 4);

  // group id, 1 warp consists of 8 groups
  int gid = lid / 4;
  int glid = lid % 4;

  half regA[8];
  half regB[4];
  float dReg[4] = {0.};

  for (int kk=0; kk < K; kk+=16) {
    // each thread loads 4 elements to shared mem
    As[ty       ][tx    ] = A[(mBlock + ty) * K + kk + tx];
    As[ty  +  16][tx    ] = A[(mBlock + ty + 16) * K + kk + tx];
    Bs[ty       ][tx    ] = B[(kk + ty) * N + nBlock + tx];
    Bs[ty       ][tx + 16] = B[(kk + ty) * N + nBlock + 16 + tx];
    __syncthreads();

    // setup registers for mma call

  }

}
