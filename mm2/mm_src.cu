#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_fp16.h>
#define N_STAGES 3


__forceinline__
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
  unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(ptxDstAddr),
      "l"(srcAddr),
      "n"(16));
}


__forceinline__
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
      :  "r"(ptxSrcAddr)
      );
}

__forceinline__
__device__ void load_matrix_x2(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(destReg[0]), "=r"(destReg[1])
      :  "r"(ptxSrcAddr)
      );
}

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

__forceinline__
__device__ void mma_m16n8k16_f16(const unsigned *A, const unsigned *B, unsigned *C, unsigned *D) {
  asm (
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "r"(C[0]), "r"(C[1])
      );
}


// Kernel 3.0: N-STAGE CP.ASYNC
__launch_bounds__(16 * 16)
__global__ void mma_matmul_3_0(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[N_STAGES*32][8];
  __shared__ uint4 Bs[N_STAGES*32][8];

  uint4 (*aLoadPtr)[8];
  uint4 (*bLoadPtr)[8];
  uint4 (*aStorePtr)[8];
  uint4 (*bStorePtr)[8];

  int blockRowStart = blockIdx.y*64;
  int blockColStart = blockIdx.x*64;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart*K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart*K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 16 * (warpID / 4);
  int warpOffsetB = 8 * (warpID % 4);

  unsigned aReg[2][8];
  unsigned bReg[2][4];
  float dReg[2][2][4] = {0.};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row/column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  const uint4 *aGlobalAddress = globalTileA + (warpID*8 + laneID/4)*K/8 + laneID%4;
  const uint4 *bGlobalAddress = globalTileB + (warpID*8 + laneID/4)*K/8 + laneID%4;

  // PRELUDE: load first (N_STAGES - 1) into shared memory
  for (int nStage=0; nStage < N_STAGES - 1; nStage++) {
    int kStart = nStage * 4;
    aStorePtr = As + 32 * nStage;
    bStorePtr = Bs + 32 * nStage;
    cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  //  MAIN LOOP OVER K BLOCKS
  for (int nStage=0; nStage < K/32; nStage++) {
    int kStart = (N_STAGES-1+nStage) * 4;
    aStorePtr = As + 32 * ((nStage + N_STAGES-1) % N_STAGES);
    bStorePtr = Bs + 32 * ((nStage + N_STAGES-1) % N_STAGES);
    aLoadPtr = As + 32 * (nStage % N_STAGES);
    bLoadPtr = Bs + 32 * (nStage % N_STAGES);

    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES-2));
    __syncthreads();

    // Preload the fragments for k=0..1, k=2..3 for both A/B tiles
    for (int m=0; m<2; m++) {
      load_matrix_x4(aReg[m]    , aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
      load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
    }
    for (int n=0; n<2; n++) {
      load_matrix_x2(bReg[n]   , bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
      load_matrix_x2(bReg[n]+ 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
    }

    // Start next cp.async
    kStart = (kStart > K/8-4) ? K/8-4 : kStart;
    cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
    asm volatile("cp.async.commit_group;\n" ::);

    // Compute the mmas
    for (int m=0; m<2; m++) {
      for (int n=0; n<2; n++) {
        mma_m16n8k16(aReg[m]    , bReg[n]    , dReg[m][n], dReg[m][n]);
        mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
      }
    }
  }

  int groupID     = laneID >> 2;
  int groupLaneID = (laneID % 4);
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n <  2; n++) {
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID    )*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      float2 *cOut2 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID + 8)*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
}

// Kernel 3.1: n stage pipeline plus 4x tiling
__launch_bounds__(16 * 16)
__global__ void mma_matmul_3_1(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[N_STAGES*64][8];
  __shared__ uint4 Bs[N_STAGES*64][8];

  uint4 (*aLoadPtr)[8];
  uint4 (*bLoadPtr)[8];
  uint4 (*aStorePtr)[8];
  uint4 (*bStorePtr)[8];

  int blockRowStart = blockIdx.y*128;
  int blockColStart = blockIdx.x*128;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart*K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart*K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 32 * (warpID / 4);
  int warpOffsetB = 16 * (warpID % 4);

  unsigned aReg[4][8];
  unsigned bReg[4][4];
  float dReg[4][4][4] = {0.};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row / column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  const uint4 *aGlobalAddress = globalTileA + (warpID*8 + laneID/4)*K/8 + laneID%4;
  const uint4 *bGlobalAddress = globalTileB + (warpID*8 + laneID/4)*K/8 + laneID%4;

  // PRELUDE: load first (N_STAGES - 1) into shared memory
  for (int nStage=0; nStage < N_STAGES - 1; nStage++) {
    int kStart = nStage * 4;
    aStorePtr = As + 64 * nStage;
    bStorePtr = Bs + 64 * nStage;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  //  MAIN LOOP OVER K BLOCKS
  for (int nStage=0; nStage < K/32; nStage++) {
    int kStart = (N_STAGES-1+nStage) * 4;
    aStorePtr = As + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    bStorePtr = Bs + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    aLoadPtr = As + 64 * (nStage % N_STAGES);
    bLoadPtr = Bs + 64 * (nStage % N_STAGES);

    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES-2));
    __syncthreads();

    // Preload the fragments for k=0..1, k=2..3 for both A/B tiles
    for (int m=0; m<4; m++) {
      load_matrix_x4(aReg[m]    , aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
      load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
    }
    for (int n=0; n<4; n++) {
      load_matrix_x2(bReg[n]    , bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
      load_matrix_x2(bReg[n] + 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
    }

    // Start next cp.async
    kStart = (kStart > K/8-4) ? K/8-4 : kStart;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);

    // Compute the mmas
    for (int m=0; m<4; m++) {
      for (int n=0; n<4; n++) {
        mma_m16n8k16(aReg[m]    , bReg[n]    , dReg[m][n], dReg[m][n]);
        mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
      }
    }
  }
  int groupID     = laneID >> 2;
  int groupLaneID = (laneID % 4);
  for (int m = 0; m < 4; m++) {
    for (int n = 0; n <  4; n++) {
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID    )*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      float2 *cOut2 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID + 8)*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
}

// Kernel 3.2: n stage pipeline plus 4x tiling, two-stage fp16/16 with fp32 accum
__launch_bounds__(16 * 16, 2)
//__maxnreg__(128)
__global__ void mma_matmul_3_2(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[N_STAGES*64][8];
  __shared__ uint4 Bs[N_STAGES*64][8];

  uint4 (*aLoadPtr)[8];
  uint4 (*bLoadPtr)[8];
  uint4 (*aStorePtr)[8];
  uint4 (*bStorePtr)[8];

  int blockRowStart = blockIdx.y*128;
  int blockColStart = blockIdx.x*128;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart*K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart*K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 32 * (warpID / 4);
  int warpOffsetB = 16 * (warpID % 4);

  unsigned aReg[4][8];
  unsigned bReg[4][4];
  unsigned cReg[2] = {0};
  unsigned dReg[4][4][2] = {0};
  half  *dRegPtr;
  float dRegAcc[4][4][4] = {0};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row / column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  const uint4 *aGlobalAddress = globalTileA + (warpID*8 + laneID/4)*K/8 + laneID%4;
  const uint4 *bGlobalAddress = globalTileB + (warpID*8 + laneID/4)*K/8 + laneID%4;

  // PRELUDE: load first (N_STAGES - 1) into shared memory
  for (int nStage=0; nStage < N_STAGES - 1; nStage++) {
    int kStart = nStage * 4;
    aStorePtr = As + 64 * nStage;
    bStorePtr = Bs + 64 * nStage;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  //  MAIN LOOP OVER K BLOCKS
  // const int accStep = 1;
  for (int nStage=0; nStage < K/32; nStage++) {
    int kStart = (N_STAGES-1+nStage) * 4;
    aStorePtr = As + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    bStorePtr = Bs + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    aLoadPtr = As + 64 * (nStage % N_STAGES);
    bLoadPtr = Bs + 64 * (nStage % N_STAGES);

    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES-2));
    __syncthreads();

    // Preload the fragments for k=0..1, k=2..3 for both A/B tiles
    for (int m=0; m<4; m++) {
      load_matrix_x4(aReg[m]    , aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
      load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
    }
    for (int n=0; n<4; n++) {
      load_matrix_x2(bReg[n]    , bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
      load_matrix_x2(bReg[n] + 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
    }

    // Start next cp.async
    kStart = (kStart > K/8-4) ? K/8-4 : kStart;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);

    // Compute the mmas
    for (int m=0; m<4; m++) {
      for (int n=0; n<4; n++) {
        dRegPtr = reinterpret_cast<half *>(dReg[m][n]);
        mma_m16n8k16_f16(aReg[m]    , bReg[n]    , cReg, dReg[m][n]);
        mma_m16n8k16_f16(aReg[m] + 4, bReg[n] + 2, cReg, dReg[m][n]);
        dRegAcc[m][n][0] += __half2float(dRegPtr[0]);
        dRegAcc[m][n][1] += __half2float(dRegPtr[1]);
        dRegAcc[m][n][2] += __half2float(dRegPtr[2]);
        dRegAcc[m][n][3] += __half2float(dRegPtr[3]);
      }
    }
  }
  int groupID     = laneID >> 2;
  int groupLaneID = (laneID % 4);
  for (int m = 0; m < 4; m++) {
    for (int n = 0; n <  4; n++) {
      float2* d0 = reinterpret_cast<float2 *>(&dRegAcc[m][n]);
      float2* d2 = reinterpret_cast<float2 *>(&dRegAcc[m][n]) + 1;
      float2 *cOut0 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID    )*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      float2 *cOut2 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID + 8)*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      *cOut0 = *d0;
      *cOut2 = *d2;
    }
  }
}


// Kernel 3.3: n stage pipeline plus 4x tiling, fp16/16 and convert output to fp32
__launch_bounds__(16 * 16)
__global__ void mma_matmul_3_3(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[N_STAGES*64][8];
  __shared__ uint4 Bs[N_STAGES*64][8];

  uint4 (*aLoadPtr)[8];
  uint4 (*bLoadPtr)[8];
  uint4 (*aStorePtr)[8];
  uint4 (*bStorePtr)[8];

  int blockRowStart = blockIdx.y*128;
  int blockColStart = blockIdx.x*128;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart*K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart*K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 32 * (warpID / 4);
  int warpOffsetB = 16 * (warpID % 4);

  unsigned aReg[4][8];
  unsigned bReg[4][4];
  unsigned dReg[4][4][2] = {0};
  half *dRegHalf;

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row / column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  const uint4 *aGlobalAddress = globalTileA + (warpID*8 + laneID/4)*K/8 + laneID%4;
  const uint4 *bGlobalAddress = globalTileB + (warpID*8 + laneID/4)*K/8 + laneID%4;

  // PRELUDE: load first (N_STAGES - 1) into shared memory
  for (int nStage=0; nStage < N_STAGES - 1; nStage++) {
    int kStart = nStage * 4;
    aStorePtr = As + 64 * nStage;
    bStorePtr = Bs + 64 * nStage;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  //  MAIN LOOP OVER K BLOCKS
  for (int nStage=0; nStage < K/32; nStage++) {
    int kStart = (N_STAGES-1+nStage) * 4;
    aStorePtr = As + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    bStorePtr = Bs + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    aLoadPtr = As + 64 * (nStage % N_STAGES);
    bLoadPtr = Bs + 64 * (nStage % N_STAGES);

    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES-2));
    __syncthreads();

    // Preload the fragments for k=0..1, k=2..3 for both A/B tiles
    for (int m=0; m<4; m++) {
      load_matrix_x4(aReg[m]    , aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
      load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
    }
    for (int n=0; n<4; n++) {
      load_matrix_x2(bReg[n]    , bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
      load_matrix_x2(bReg[n] + 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
    }

    // Start next cp.async
    kStart = (kStart > K/8-4) ? K/8-4 : kStart;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);

    // Compute the mmas
    for (int m=0; m<4; m++) {
      for (int n=0; n<4; n++) {
        mma_m16n8k16_f16(aReg[m]    , bReg[n]    , dReg[m][n], dReg[m][n]);
        mma_m16n8k16_f16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
      }
    }
  }
  int groupID     = laneID >> 2;
  int groupLaneID = (laneID % 4);
  for (int m = 0; m < 4; m++) {
    for (int n = 0; n <  4; n++) {
      dRegHalf = reinterpret_cast<half *>(dReg[m][n]);
      float2 d0 = make_float2(__half2float(dRegHalf[0]), __half2float(dRegHalf[1]));
      float2 d2 = make_float2(__half2float(dRegHalf[2]), __half2float(dRegHalf[3]));
      float2 *cOut0 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID    )*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      float2 *cOut2 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID + 8)*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
}

// Kernel 3.4: n stage pipeline plus 4x tiling, customizable two-stage fp16/16 with fp32 accum every accStep steps
__launch_bounds__(16 * 16, 2)
//__maxnreg__(128)
__global__ void mma_matmul_3_4(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[N_STAGES*64][8];
  __shared__ uint4 Bs[N_STAGES*64][8];

  uint4 (*aLoadPtr)[8];
  uint4 (*bLoadPtr)[8];
  uint4 (*aStorePtr)[8];
  uint4 (*bStorePtr)[8];

  int blockRowStart = blockIdx.y*128;
  int blockColStart = blockIdx.x*128;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart*K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart*K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 32 * (warpID / 4);
  int warpOffsetB = 16 * (warpID % 4);

  unsigned aReg[4][8];
  unsigned bReg[4][4];
  float dReg[4][4][4] = {0.};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row / column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  const uint4 *aGlobalAddress = globalTileA + (warpID*8 + laneID/4)*K/8 + laneID%4;
  const uint4 *bGlobalAddress = globalTileB + (warpID*8 + laneID/4)*K/8 + laneID%4;

  // PRELUDE: load first (N_STAGES - 1) into shared memory
  for (int nStage=0; nStage < N_STAGES - 1; nStage++) {
    int kStart = nStage * 4;
    aStorePtr = As + 64 * nStage;
    bStorePtr = Bs + 64 * nStage;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  //  MAIN LOOP OVER K BLOCKS
  for (int nStage=0; nStage < K/32; nStage++) {
    int kStart = (N_STAGES-1+nStage) * 4;
    aStorePtr = As + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    bStorePtr = Bs + 64 * ((nStage + N_STAGES-1) % N_STAGES);
    aLoadPtr = As + 64 * (nStage % N_STAGES);
    bLoadPtr = Bs + 64 * (nStage % N_STAGES);

    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES-2));
    __syncthreads();

    // Preload the fragments for k=0..1, k=2..3 for both A/B tiles
    for (int m=0; m<4; m++) {
      load_matrix_x4(aReg[m]    , aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
      load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
    }
    for (int n=0; n<4; n++) {
      load_matrix_x2(bReg[n]    , bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
      load_matrix_x2(bReg[n] + 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
    }

    // Start next cp.async
    kStart = (kStart > K/8-4) ? K/8-4 : kStart;
    cp_async(aStorePtr[storeRow     ] + storeCol, aGlobalAddress + kStart);
    cp_async(aStorePtr[storeRow + 32] + storeCol, aGlobalAddress + 64 * K/8 + kStart);
    cp_async(bStorePtr[storeRow     ] + storeCol, bGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow + 32] + storeCol, bGlobalAddress + 64 * K/8 + kStart);
    asm volatile("cp.async.commit_group;\n" ::);

    // Compute the mmas
    for (int m=0; m<4; m++) {
      for (int n=0; n<4; n++) {
        mma_m16n8k16(aReg[m]    , bReg[n]    , dReg[m][n], dReg[m][n]);
        mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n], dReg[m][n]);
      }
    }
  }
  int groupID     = laneID >> 2;
  int groupLaneID = (laneID % 4);
  for (int m = 0; m < 4; m++) {
    for (int n = 0; n <  4; n++) {
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID    )*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      float2 *cOut2 = reinterpret_cast<float2 *>(&C[(blockRowStart + m*16 + 2 * warpOffsetA + groupID + 8)*N + blockColStart + n*8 + 2 * warpOffsetB + 2*groupLaneID]);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
}

torch::Tensor mm_tc_3_0(torch::Tensor A, torch::Tensor Bcol) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(Bcol.is_cuda(), "Bcol must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(Bcol.dtype() == torch::kFloat16, "Bcol must be fp16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(Bcol.is_contiguous(), "Bcol must be contiguous (row-major)");
  TORCH_CHECK(A.dim() == 2 && Bcol.dim() == 2, "A and Bcol must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(Bcol.size(1) == K, "Bcol must have shape [N,K] with same K as A");
  int64_t N = Bcol.size(0);

  TORCH_CHECK((M % 64) == 0, "M must be multiple of 64");
  TORCH_CHECK((N % 64) == 0, "N must be multiple of 64");
  TORCH_CHECK((K % 32)  == 0, "K must be multiple of 32");

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

  constexpr int THREADS = 256;

  dim3 block(THREADS);
  dim3 grid((unsigned)(N / 64), (unsigned)(M / 64));

  mma_matmul_3_0<<<grid, block>>>(
      (half*)A.data_ptr<at::Half>(),
      (half*)Bcol.data_ptr<at::Half>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }
  return C;
}


torch::Tensor mm_tc_3_1(torch::Tensor A, torch::Tensor Bcol) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(Bcol.is_cuda(), "Bcol must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(Bcol.dtype() == torch::kFloat16, "Bcol must be fp16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(Bcol.is_contiguous(), "Bcol must be contiguous (row-major)");
  TORCH_CHECK(A.dim() == 2 && Bcol.dim() == 2, "A and Bcol must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(Bcol.size(1) == K, "Bcol must have shape [N,K] with same K as A");
  int64_t N = Bcol.size(0);

  TORCH_CHECK((M % 128) == 0, "M must be multiple of 128");
  TORCH_CHECK((N % 128) == 0, "N must be multiple of 128");
  TORCH_CHECK((K % 32)  == 0, "K must be multiple of 32");

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

  constexpr int THREADS = 256;

  dim3 block(THREADS);
  dim3 grid((unsigned)(N / 128), (unsigned)(M / 128));

  mma_matmul_3_1<<<grid, block>>>(
      (half*)A.data_ptr<at::Half>(),
      (half*)Bcol.data_ptr<at::Half>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }
  return C;
}

torch::Tensor mm_tc_3_2(torch::Tensor A, torch::Tensor Bcol) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(Bcol.is_cuda(), "Bcol must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(Bcol.dtype() == torch::kFloat16, "Bcol must be fp16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(Bcol.is_contiguous(), "Bcol must be contiguous (row-major)");
  TORCH_CHECK(A.dim() == 2 && Bcol.dim() == 2, "A and Bcol must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(Bcol.size(1) == K, "Bcol must have shape [N,K] with same K as A");
  int64_t N = Bcol.size(0);

  // Size constraints for our specialized kernel
  TORCH_CHECK((M % 128) == 0, "M must be multiple of 128");
  TORCH_CHECK((N % 128) == 0, "N must be multiple of 128");
  TORCH_CHECK((K % 32)  == 0, "K must be multiple of 32");

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

  constexpr int THREADS = 256;

  dim3 block(THREADS);
  dim3 grid((unsigned)(N / 128), (unsigned)(M / 128));

  mma_matmul_3_2<<<grid, block>>>(
      (half*)A.data_ptr<at::Half>(),
      (half*)Bcol.data_ptr<at::Half>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }
  return C;
}

torch::Tensor mm_tc_3_3(torch::Tensor A, torch::Tensor Bcol) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(Bcol.is_cuda(), "Bcol must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(Bcol.dtype() == torch::kFloat16, "Bcol must be fp16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(Bcol.is_contiguous(), "Bcol must be contiguous (row-major)");
  TORCH_CHECK(A.dim() == 2 && Bcol.dim() == 2, "A and Bcol must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(Bcol.size(1) == K, "Bcol must have shape [N,K] with same K as A");
  int64_t N = Bcol.size(0);

  // Size constraints for our specialized kernel
  TORCH_CHECK((M % 128) == 0, "M must be multiple of 128");
  TORCH_CHECK((N % 128) == 0, "N must be multiple of 128");
  TORCH_CHECK((K % 32)  == 0, "K must be multiple of 32");

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

  constexpr int THREADS = 256;

  dim3 block(THREADS);
  dim3 grid((unsigned)(N / 128), (unsigned)(M / 128));

  mma_matmul_3_3<<<grid, block>>>(
      (half*)A.data_ptr<at::Half>(),
      (half*)Bcol.data_ptr<at::Half>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }
  return C;
}

torch::Tensor mm_tc_3_4(torch::Tensor A, torch::Tensor Bcol) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA");
  TORCH_CHECK(Bcol.is_cuda(), "Bcol must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(Bcol.dtype() == torch::kFloat16, "Bcol must be fp16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(Bcol.is_contiguous(), "Bcol must be contiguous (row-major)");
  TORCH_CHECK(A.dim() == 2 && Bcol.dim() == 2, "A and Bcol must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(Bcol.size(1) == K, "Bcol must have shape [N,K] with same K as A");
  int64_t N = Bcol.size(0);

  // Size constraints for our specialized kernel
  TORCH_CHECK((M % 128) == 0, "M must be multiple of 128");
  TORCH_CHECK((N % 128) == 0, "N must be multiple of 128");
  TORCH_CHECK((K % 32)  == 0, "K must be multiple of 32");

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

  constexpr int THREADS = 256;

  dim3 block(THREADS);
  dim3 grid((unsigned)(N / 128), (unsigned)(M / 128));

  mma_matmul_3_4<<<grid, block>>>(
      (half*)A.data_ptr<at::Half>(),
      (half*)Bcol.data_ptr<at::Half>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }
  return C;
}

TORCH_LIBRARY(lib, m) {
  m.def("mm_3_1(Tensor A, Tensor Bcol) -> Tensor");
  m.impl("mm_3_1", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(mm_tc_3_1)));
  m.def("mm_3_0(Tensor A, Tensor Bcol) -> Tensor");
  m.impl("mm_3_0", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(mm_tc_3_0)));
  m.def("mm_3_4(Tensor A, Tensor Bcol) -> Tensor");
  m.impl("mm_3_4", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(mm_tc_3_4)));
  m.def("mm_3_2(Tensor A, Tensor Bcol) -> Tensor");
  m.impl("mm_3_2", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(mm_tc_3_2)));
  m.def("mm_3_3(Tensor A, Tensor Bcol) -> Tensor");
  m.impl("mm_3_3", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(mm_tc_3_3)));
}