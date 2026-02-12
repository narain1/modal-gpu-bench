#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <float.h>

void initialize_random_normal(float *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_min(float v) {
    __shared__ float warp_mins[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    v = warp_reduce_min(v);
    if (lane == 0) warp_mins[wid] = v;
    __syncthreads();

    v = (tid < blockDim.x / 32) ? warp_mins[lane] : FLT_MAX;
    if (wid == 0) v = warp_reduce_min(v);
    return v;
}


template <unsigned int threads_per_block, unsigned int batch_size_vectors>
__global__ void __launch_bounds__(threads_per_block) reduction_kernel_batched(const float* __restrict__ inp, float* __restrict__ out, const size_t N) {
    const float4 *inp4 = reinterpret_cast<const float4*>(inp);
    
    // Each block processes (threads_per_block * batch_size_vectors) float4 elements
    // The grid size is calculated to cover N
    
    size_t block_offset = blockIdx.x * (threads_per_block * batch_size_vectors);
    size_t idx = block_offset + threadIdx.x;

    // 4 Accumulators
    float min1 = FLT_MAX;
    float min2 = FLT_MAX;
    float min3 = FLT_MAX;
    float min4 = FLT_MAX;
    
    #pragma unroll
    for (int i = 0; i < batch_size_vectors; ++i) {
        if (idx < N/4) // REMOVED BOUNDS CHECK FOR SPEED - Ensure N is large enough or pad
        {
             float4 v = __ldg(&inp4[idx]);
             min1 = min(min1, v.x);
             min2 = min(min2, v.y);
             min3 = min(min3, v.z);
             min4 = min(min4, v.w);
             idx += threads_per_block;
        }
    }
    
    float min_val = min(min1, min2);
    min_val = min(min_val, min3);
    min_val = min(min_val, min4);
    min_val = block_reduce_min(min_val);

    if (threadIdx.x == 0) {
        atomicMin((int*)out, __float_as_int(min_val));
    }
}

void verify_reduction(float *input, float *output, const size_t n) {
    double cpu_min = FLT_MAX;
    #pragma unroll
    for (size_t i = 0; i < n; ++i) {
        cpu_min = min(cpu_min, static_cast<double>(input[i]));
    }
    printf("kernel is valid, %s\n", abs(cpu_min - *output) < 1e-1 ? "true" : "false");
    printf("CPU Min: %f\n", cpu_min);
    printf("GPU Min: %f\n", *output);
}

void launch_reduction(float *input, float *output, const size_t n) {
    constexpr int threads_per_block = 256;
    constexpr int batch_size_vectors = 8; 
    
    size_t vectors_total = n / 4;
    size_t vectors_per_block = threads_per_block * batch_size_vectors;
    size_t numBlocks = (vectors_total + vectors_per_block - 1) / vectors_per_block;
    
    reduction_kernel_batched<threads_per_block, batch_size_vectors><<<numBlocks, threads_per_block>>>(input, output, n);
}

int main() {
    size_t N = 1 << 24;
    float *h_in, *h_out;
    float *d_in, *d_out;

    h_in = (float *)malloc(N * sizeof(float));
    h_out = (float *)malloc(sizeof(float));
    initialize_random_normal(h_in, N);

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // warmup 
    for (int i = 0; i < 10; ++i) {
        launch_reduction(d_in, d_out, N);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        launch_reduction(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Average time for reduction: %f ms\n", milliseconds / 100.0);
    //tflops
    double tflops = static_cast<double>(N) * 2.0 / (
        milliseconds / 100.0 / 1e3) / 1e12;
    printf("TFLOPS: %f\n", tflops);

    double bytes = double(N) * sizeof(float);
    double bandwidth = bytes / (milliseconds / 100.0 / 1e3) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    cudaMemset(d_out, 0, sizeof(float));
    launch_reduction(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("the min is %f\n", h_out[0]);
    verify_reduction(h_in, h_out, N);


    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);

    return 0;
}
