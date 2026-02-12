#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

struct SumHalfFloat {
    __device__ __forceinline__ float operator()(const float &a, const float &b) const {
        return a + b;
    }
    __device__ __forceinline__ float operator()(const float &a, const __half &b) const {
        return a + __half2float(b);
    }
    __device__ __forceinline__ float operator()(const __half &a, const float &b) const {
        return __half2float(a) + b;
    }
    __device__ __forceinline__ float operator()(const __half &a, const __half &b) const {
        return __half2float(a) + __half2float(b);
    }
};

void initialize_random(__half *d_in, size_t n) {
    std::vector<__half> h_in(n);
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        h_in[i] = __float2half(dis(gen));
    }
    cudaMemcpy(d_in, h_in.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

int main() {
    size_t N = 1 << 24; // 16M elements
    __half *d_in;
    float *d_out;
    
    cudaMalloc(&d_in, N * sizeof(__half));
    cudaMalloc(&d_out, sizeof(float));

    initialize_random(d_in, N);
    
    SumHalfFloat reduction_op;
    float init = 0.0f;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, N, reduction_op, init);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Warmup
    for (int i = 0; i < 10; i++) {
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, N, reduction_op, init);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, N, reduction_op, init);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemset(d_out, 0, sizeof(float));
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, N, reduction_op, init);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / 100.0f;

    printf("Average time for CUB reduction: %f ms\n", avg_ms);
    
    double bytes = (double)N * sizeof(__half);
    double bandwidth = bytes / (avg_ms / 1000.0) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    // Verify output
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU Sum: %f\n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp_storage);
    
    return 0;
}
