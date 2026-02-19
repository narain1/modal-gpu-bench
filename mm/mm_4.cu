#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cublas_v2.h>
#include <cub/cub.cuh>


void initialize_random_normal(float *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}

template <typename T> struct Afrag_16x16 {
  static constexpr size_t ne = 8; // num of elements per thread

  T x[ne];

  static __device__ size_t get_row(int tid, int l) {
    int group_id = tid >> 2;
    return group_id + 8 * ((l / 2) % 2);
  }

  static __device__ size_t get_col(int tid, int l) {
    return 2 * (tid % 4) + (l % 2) + 8 * (l / 4);
  }
};

template <typename T> struct Bfrag_16x8 {
  static constexpr size_t ne = 4;
  T x[ne] = {};
  static __device__ size_t get_row(int tid, int l) {
    return (tid % 4) * 2 + (l % 2) + 8 * (l / 2);
  }

  static __device__ size_t get_col(int tid, int l) { return tid >> 2; }
};

// BM, BN: Block tile dimensions (shared memory tiles)
// BK: K dimension of block tile
// TM, TN: Thread tile dimensions (each thread computes TM x TN outputs)
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadCol = tx * TN;
    const int threadRow = ty * TM;
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Global position of the thread's tile
    const int globalRow = by * BM + threadRow;
    const int globalCol = bx * BN + threadCol;
    
    // Number of threads per block dimension
    const int threadsPerBlockX = BN / TN;
    const int threadsPerBlockY = BM / TM;
    
    // Each thread computes TM x TN output elements - use register array
    float accum[TM][TN] = {0.0f};
    
    // Register caches for As and Bs tiles
    float regA[TM];
    float regB[TN];
    
    // Number of tiles needed
    const int numTiles = (K + BK - 1) / BK;
    
    // Loop over tiles in K dimension
    for (int t = 0; t < numTiles; ++t) {
        // Collaborative loading of A tile into shared memory
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < BM; i += threadsPerBlockY) {
            #pragma unroll
            for (int j = 0; j < BK; j += threadsPerBlockX) {
                int row = i + ty;
                int col = j + tx;
                if (row < BM && col < BK) {
                    int globalRowA = by * BM + row;
                    int globalColA = t * BK + col;
                    As[row][col] = (globalRowA < M && globalColA < K) ? 
                                   A[globalRowA * K + globalColA] : 0.0f;
                }
            }
        }
        
        // Collaborative loading of B tile into shared memory
        #pragma unroll
        for (int i = 0; i < BK; i += threadsPerBlockY) {
            #pragma unroll
            for (int j = 0; j < BN; j += threadsPerBlockX) {
                int row = i + ty;
                int col = j + tx;
                if (row < BK && col < BN) {
                    int globalRowB = t * BK + row;
                    int globalColB = bx * BN + col;
                    Bs[row][col] = (globalRowB < K && globalColB < N) ? 
                                   B[globalRowB * N + globalColB] : 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial products for this tile
        // Each thread computes TM x TN outputs
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load TM elements from As into registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[threadRow + i][k];
            }
            
            // Load TN elements from Bs into registers
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[k][threadCol + j];
            }
            
            // Outer product: update TM x TN accumulators
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] = fmaf(regA[i], regB[j], accum[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = globalRow + i;
            int col = globalCol + j;
            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}


__global__ void compute_abs_diff(const float *a, const float *b, float *diff, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        diff[idx] = fabsf(a[idx] - b[idx]);
    }
}




int main() {
    int m = 2048, n = 2048, k = 2048;
    float *A, *B;
    float *d_A, *d_B, *d_C;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);
    A = (float*)malloc(size_A);
    B = (float*)malloc(size_B);
    initialize_random_normal(A, m * k);
    initialize_random_normal(B, k * n);
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    // Block tile dimensions (shared memory)
    const int BM = 64;  // Increased from 16
    const int BN = 64;  // Increased from 16
    const int BK = 8;    // K tile dimension
    
    // Thread tile dimensions (each thread computes TM x TN outputs)
    const int TM = 4;
    const int TN = 4;
    
    // Threads per block = (BM/TM) x (BN/TN)
    dim3 threadsPerBlock(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 numBlocks((n + BN - 1) / BN,
                   (m + BM - 1) / BM);
    for (int i = 0; i < 10; ++i) {
        matrix_multiply<BM, BN, BK, TM, TN><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        matrix_multiply<BM, BN, BK, TM, TN><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_ms = milliseconds / 100.0;

    // Verify with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_C_ref;
    cudaMalloc(&d_C_ref, size_C);
    cudaMemset(d_C_ref, 0, size_C);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k,
                &alpha,
                d_B, n,
                d_A, k,
                &beta,
                d_C_ref, n);

    // Compute absolute differences on GPU
    float *d_diff;
    cudaMalloc(&d_diff, size_C);
    int total_elements = m * n;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    compute_abs_diff<<<blocks, threads>>>(d_C, d_C_ref, d_diff, total_elements);
    
    // Use CUB to find maximum error
    float *d_max_error;
    cudaMalloc(&d_max_error, sizeof(float));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary storage requirements
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_diff, d_max_error, total_elements);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Find maximum error
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_diff, d_max_error, total_elements);
    
    float max_error;
    cudaMemcpy(&max_error, d_max_error, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    const float tolerance = 1e-3;
    if (max_error <= tolerance) {
        printf("Results are correct! Max error: %e\n", max_error);
    } else {
        printf("Results are incorrect! Max error: %e (tolerance: %e)\n", max_error, tolerance);
    }

    cudaFree(d_temp_storage);
    cudaFree(d_max_error);
    cudaFree(d_diff);
    cublasDestroy(handle);
    cudaFree(d_C_ref);
    cudaDeviceSynchronize();

    std::cout << "average time: " << avg_ms << " ms" << std::endl;
    double bytes = (double)(m * k + n * k + m * n) * sizeof(float);
    double bandwidth = bytes / (avg_ms * 1e-3) / 1e9;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    double tflops = static_cast<double>(2) * m * n * k / (avg_ms * 1e-3) / 1e12;
    std::cout << "TFLOPS: " << tflops << std::endl;
    
    return 0;
}
