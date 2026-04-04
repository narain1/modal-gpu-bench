#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define BLOCK_SIZE 16


void initialize_random_normal(float *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}


__global__ void matrix_multiply_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void compute_abs_diff(const float *a, const float *b, float *diff, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        diff[idx] = fabsf(a[idx] - b[idx]);
    }
}


int main() {
    int m = 1024, n = 1024, k = 1024;
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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(CEIL_DIV(m, threadsPerBlock.x),
                   CEIL_DIV(n, threadsPerBlock.y));
    for (int i = 0; i < 10; ++i) {
        matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Average time for matrix multiplication: %f ms\n", milliseconds / 100.0);
    double tflops = static_cast<double>(2) * m * n * k / (milliseconds / 100.0 / 1e3) / 1e12;
    printf("TFLOPS: %f\n", tflops);

    double bytes = (double(m)*k + double(k)*n + double(m)*n) * sizeof(float);
    double bandwidth = bytes / (milliseconds / 100.0 / 1e3) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    // Verify with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_C_ref;
    cudaMalloc(&d_C_ref, size_C);
    cudaMemset(d_C_ref, 0, size_C);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLAS uses column-major, so compute C^T = B^T * A^T (which equals C = A * B in row-major)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k,
                &alpha,
                d_B, n,
                d_A, k,
                &beta,
                d_C_ref, n);
    
    // Compute differences on GPU
    float *d_diff;
    cudaMalloc(&d_diff, size_C);
    int total_elements = m * n;
    compute_abs_diff<<<CEIL_DIV(total_elements, 256), 256>>>(d_C, d_C_ref, d_diff, total_elements);
    
    // Use CUB to find max error
    float *d_max_error;
    cudaMalloc(&d_max_error, sizeof(float));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_diff, d_max_error, total_elements);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_diff, d_max_error, total_elements);
    
    float max_diff;
    cudaMemcpy(&max_diff, d_max_error, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Max absolute difference: %e\n", max_diff);
    bool correct = max_diff < 1e-3f;

    if (correct) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect!\n");
    }

    cudaFree(d_temp_storage);
    cudaFree(d_max_error);
    cudaFree(d_diff);
    cudaFree(d_C_ref);
    cublasDestroy(handle);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
}
