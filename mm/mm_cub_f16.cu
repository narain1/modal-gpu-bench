#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>


void initialize_random_normal(__half *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = __half(distribution(generator));
    }
}

int main() {
    int m = 4096, n = 4096, k = 4096;
    __half *A, *B;
    __half *d_A, *d_B, *d_C;
    size_t size_A = m * k * sizeof(__half);
    size_t size_B = k * n * sizeof(__half);
    size_t size_C = m * n * sizeof(__half);
    A = (__half*)malloc(size_A);
    B = (__half*)malloc(size_B);
    initialize_random_normal(A, m * k);
    initialize_random_normal(B, k * n);
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    cudaEvent_t start, stop;

    // Warm up
    for (int i = 0; i < 10; ++i) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Average time for matrix multiplication: %f ms\n", milliseconds / 100.0);
    double tflops = static_cast<double>(2) * m * n * k / (milliseconds / 100.0 / 1e3) / 1e12;
    printf("TFLOPS: %f\n", tflops);
    double bytes = (double(m)*k + double(k)*n + double(m)*n) * sizeof(__half); 
    double bandwidth = bytes / (milliseconds / 1e3) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
}