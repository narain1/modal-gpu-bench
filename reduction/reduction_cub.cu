#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

void initialize_random(float *d_in, size_t n) {
    std::vector<float> h_in(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        h_in[i] = dis(gen);
    }
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

int main() {
    size_t N = 1 << 24; // 16M elements
    float *d_in;
    float *d_out;
    
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    // Initialize with data
    initialize_random(d_in, N);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    // Request the amount of temporary storage needed
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Warmup
    for (int i = 0; i < 10; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / 100.0f;

    printf("Average time for CUB reduction: %f ms\n", avg_ms);
    
    double bytes = (double)N * sizeof(float);
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
