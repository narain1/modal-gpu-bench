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
    int row = blockIdx.y * bm + threadIdx.y;
    int col = blockIdx.x * bn + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + bk - 1) / bk; ++t) {
        // loading to smem
        As[threadIdx.y][threadIdx.x] = (row < M && t * bk + threadIdx.x < K) ? A[row * K + t * bk + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t * bk + threadIdx.y < K) ? B[(t * bk + threadIdx.y) * N + col] : 0.0f;

        __syncthreads();
        // compute the sum along a column of bk
        for (int k = 0; k < bk; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
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

    const int BLOCKSIZE = 16;
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 numBlocks((n + BLOCKSIZE - 1) / BLOCKSIZE,
                   (m + BLOCKSIZE - 1) / BLOCKSIZE);
    for (int i = 0; i < 10; ++i) {
        matrix_multiply<BLOCKSIZE, BLOCKSIZE, BLOCKSIZE><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        matrix_multiply<BLOCKSIZE, BLOCKSIZE, BLOCKSIZE><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }

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
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_ms = milliseconds / 100.0;
    std::cout << "Time: " << avg_ms << " ms" << std::endl;
    double bytes = (double)(m * k + n * k + m * n) * sizeof(float);
    double bandwidth = bytes / (avg_ms * 1e-3) / 1e9;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    double tflops = static_cast<double>(2) * m * n * k / (avg_ms * 1e-3) / 1e12;
    std::cout << "TFLOPS: " << tflops << std::endl;
    
    return 0;
}
