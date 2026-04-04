#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <random>
#include <iostream>

#define BLOCKSIZE 16

void initialize_random_normal(float *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}


__global__ void matrix_multiply_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < m && y < n) {
        float tmp = 0.0f;
        for (int i = 0; i < k; ++i) {
            float _a, _b;
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(_a) : "l"(a + x * k + i));
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(_b) : "l"(b + i * n + y));
            asm("fma.rn.f32 %0, %1, %2, %0;" : "+f"(tmp) : "f"(_a), "f"(_b));
        }
        asm volatile("st.global.f32 [%0], %1;" :: "l"(c + x * n + y), "f"(tmp));
    }
}


__global__ void compute_abs_diff(const float *a, const float *b, float *diff, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float _a, _b;
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(_a) : "l"(a + idx));
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(_b) : "l"(b + idx));
        asm volatile("st.global.f32 [%0], %1;" :: "l"(diff + idx), "f"(fabsf(_a - _b)));
    }
}


int main() {
    int m = 4096, n = 4096, k = 4096;
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

    dim3 threadsPerBlock(BLOCKSIZE * BLOCKSIZE);
    dim3 numBlocks((m + BLOCKSIZE - 1) / BLOCKSIZE,
                   (n + BLOCKSIZE - 1) / BLOCKSIZE);
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
    
    // cuBLAS uses column-major order, so we compute C^T = B^T * A^T
    // which is equivalent to C = A * B in row-major
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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
}
