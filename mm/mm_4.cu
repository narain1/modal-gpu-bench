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


template<const int bm, const int bn, const int bk>
__global__ void matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[bm][bk];
    __shared__ float Bs[bk][bn];
    
    constexpr int V = 4; // vector width for loading/storing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * bm + ty;
    int colBase = bx * bn + tx * V;
    
    float sum[V] = {0.0f};
    
    int numTiles = (K + bk - 1) / bk;
    
    for (int t = 0; t < numTiles; ++t) {
        // Each thread loads one element
        {
            int tid = ty * (bn / V) + tx;  // flat thread id
            int totalLoads = (bm * bk) / V;
            
            for (int i = tid; i < totalLoads; i += (bm * (bn / V))) {
                int aSmRow = i / (bk / V);
                int aSmCol = (i % (bk / V)) * V;

                int aRow = by * bm + aSmRow;
                int aCol = t * bk + aSmCol;

                float4 aVec;
                if (aRow < M && aCol + V - 1 < K) {
                    aVec = *reinterpret_cast<const float4*>(&A[aRow * K + aCol]);
                } else {
                    aVec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    // Handle partial boundary
                    if (aRow < M) {
                        if (aCol + 0 < K) aVec.x = A[aRow * K + aCol + 0];
                        if (aCol + 1 < K) aVec.y = A[aRow * K + aCol + 1];
                        if (aCol + 2 < K) aVec.z = A[aRow * K + aCol + 2];
                        if (aCol + 3 < K) aVec.w = A[aRow * K + aCol + 3];
                    }
                }

                As[aSmRow][aSmCol + 0] = aVec.x;
                As[aSmRow][aSmCol + 1] = aVec.y;
                As[aSmRow][aSmCol + 2] = aVec.z;
                As[aSmRow][aSmCol + 3] = aVec.w;
            }
        }
        
        // Load tile of B into shared memory
        {
            int tid = ty * (bn / V) + tx;
            int totalLoads = (bk * bn) / V;

            for (int i = tid; i < totalLoads; i += (bm * (bn / V))) {
                int bSmRow = i / (bn / V);
                int bSmCol = (i % (bn / V)) * V;

                int bRow = t * bk + bSmRow;
                int bCol = bx * bn + bSmCol;

                float4 bVec;
                if (bRow < K && bCol + V - 1 < N) {
                    bVec = *reinterpret_cast<const float4*>(&B[bRow * N + bCol]);
                } else {
                    bVec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (bRow < K) {
                        if (bCol + 0 < N) bVec.x = B[bRow * N + bCol + 0];
                        if (bCol + 1 < N) bVec.y = B[bRow * N + bCol + 1];
                        if (bCol + 2 < N) bVec.z = B[bRow * N + bCol + 2];
                        if (bCol + 3 < N) bVec.w = B[bRow * N + bCol + 3];
                    }
                }

                Bs[bSmRow][bSmCol + 0] = bVec.x;
                Bs[bSmRow][bSmCol + 1] = bVec.y;
                Bs[bSmRow][bSmCol + 2] = bVec.z;
                Bs[bSmRow][bSmCol + 3] = bVec.w;
            }
        }

        __syncthreads();
        
        // Compute partial product for this tile
        for (int i = 0; i < bk; ++i) {
            float aVal = As[ty][i];
            sum[0] += aVal * Bs[i][tx * V + 0];
            sum[1] += aVal * Bs[i][tx * V + 1];
            sum[2] += aVal * Bs[i][tx * V + 2];
            sum[3] += aVal * Bs[i][tx * V + 3];
        }

        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M) {
        for (int v = 0; v < V; ++v) {
            if (colBase + v < N) {
                C[row * N + colBase + v] = sum[v];
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

    const int BM = 16;
    const int BN = 16;
    const int BK = 16;
    const int V = 4;
    dim3 threadsPerBlock(BN/V, BM);
    dim3 numBlocks((n + BN - 1) / BN,
                   (m + BM - 1) / BM);
    for (int i = 0; i < 10; ++i) {
        matrix_multiply<BM, BN, BK><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        matrix_multiply<BM, BN, BK><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
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
