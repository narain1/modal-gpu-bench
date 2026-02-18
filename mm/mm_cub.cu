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

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEvent_t start, stop;

    // Warm up
    for (int i = 0; i < 10; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / 100.0f;
    printf("Average time per SGEMM: %.6f ms\n", avg_ms);

    double bytes = (double)(m * k + n * k + m * n) * sizeof(float);
    double bandwidth = bytes / (avg_ms * 1e-3) / 1e9;

    printf("Bandwidth: %.6f GB/s\n", bandwidth);
    double tflops = static_cast<double>(2) * m * n * k / (avg_ms * 1e-3) / 1e12;
    printf("TFLOPS: %.6f\n", tflops);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
}