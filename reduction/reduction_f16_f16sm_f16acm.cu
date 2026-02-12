#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

void initialize_random_normal(__half *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<__half> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = __float2half(distribution(generator));
    }
    cudaMemcpy(data, h_data.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp reduce that accepts fp16 and accumulates in fp32
__device__ __forceinline__ float warp_reduce_sum_fp16(__half val) {
    float sum = __half2float(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset); // warp reduction on fp16 values
        sum += __half2float(other);
    }
    return sum;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    v = (tid < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

// Block reduce that accepts fp16 and accumulates in fp32
__device__ __forceinline__ float block_reduce_sum_fp16(__half v) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    float warp_sum = warp_reduce_sum_fp16(v);
    if (lane == 0) warp_sums[wid] = warp_sum;
    __syncthreads();

    float v_final = (tid < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    if (wid == 0) v_final = warp_reduce_sum(v_final);
    return v_final;
}


template <unsigned int threads_per_block, unsigned int batch_size_vectors>
__global__ void __launch_bounds__(threads_per_block) reduction_kernel_batched(const __half* __restrict__ inp, float* __restrict__ out, const size_t N) {
    const __half2 *inp2 = reinterpret_cast<const __half2*>(inp);

    // Each block processes (threads_per_block * batch_size_vectors) __half2 elements
    // But each thread processes 4 __half2 elements (8 fp16 values) per iteration

    size_t block_offset = blockIdx.x * (threads_per_block * batch_size_vectors * 4); // 4x more elements per thread
    size_t idx = block_offset + threadIdx.x;

    // 8 fp16 accumulators in registers
    __half sum1 = __float2half(0.0f), sum2 = __float2half(0.0f);
    __half sum3 = __float2half(0.0f), sum4 = __float2half(0.0f);
    __half sum5 = __float2half(0.0f), sum6 = __float2half(0.0f);
    __half sum7 = __float2half(0.0f), sum8 = __float2half(0.0f);

    #pragma unroll
    for (int i = 0; i < batch_size_vectors; ++i) {
        if ((idx + 3 * threads_per_block) < N/2) // Check we can load 4 __half2 elements safely
        {
             // Load 4 __half2 elements with proper stride for coalesced access
             __half2 v1 = __ldg(&inp2[idx]);
             __half2 v2 = __ldg(&inp2[idx + threads_per_block]);
             __half2 v3 = __ldg(&inp2[idx + 2 * threads_per_block]);
             __half2 v4 = __ldg(&inp2[idx + 3 * threads_per_block]);

             // Keep values as fp16 in registers
             sum1 = __hadd(sum1, v1.x);
             sum2 = __hadd(sum2, v1.y);
             sum3 = __hadd(sum3, v2.x);
             sum4 = __hadd(sum4, v2.y);
             sum5 = __hadd(sum5, v3.x);
             sum6 = __hadd(sum6, v3.y);
             sum7 = __hadd(sum7, v4.x);
             sum8 = __hadd(sum8, v4.y);

             idx += 4 * threads_per_block; // Move to next set of 4 __half2 elements
        }
    }

    // Accumulate fp16 values then reduce with fp32 precision
    __half h_sum_pair1 = __hadd(__hadd(sum1, sum2), __hadd(sum3, sum4));
    __half h_sum_pair2 = __hadd(__hadd(sum5, sum6), __hadd(sum7, sum8));
    __half h_sum = __hadd(h_sum_pair1, h_sum_pair2);

    // Shuffle fp16 values and accumulate in fp32
    float sum = block_reduce_sum_fp16(h_sum);

    if (threadIdx.x == 0) {
        atomicAdd(&out[0], sum);
    }
}

void verify_reduction(__half *input, float *output, const size_t n) {
    double cpu_sum = 0.0;
    std::vector<__half> h_input(n);
    cudaMemcpy(h_input.data(), input, n * sizeof(__half), cudaMemcpyDeviceToHost);
    #pragma unroll
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += __half2float(h_input[i]);
    }
    printf("kernel is valid, %s\n", abs(cpu_sum - *output) < 1e-1 ? "true" : "false");
    printf("CPU Sum: %f\n", cpu_sum);
    printf("GPU Sum: %f\n", *output);
}

void launch_reduction(__half *input, float *output, const size_t n) {
    constexpr int threads_per_block = 256;
    constexpr int batch_size_vectors = 8;

    // Each thread processes 4 __half2 elements (8 fp16 values) per batch iteration
    size_t elements_per_block = threads_per_block * batch_size_vectors * 8; // 8 fp16 elements per thread per batch
    size_t numBlocks = (n + elements_per_block - 1) / elements_per_block;

    reduction_kernel_batched<threads_per_block, batch_size_vectors><<<numBlocks, threads_per_block>>>(input, output, n);
}

int main() {
    size_t N = 1 << 24;
    __half *d_in;
    float *h_out, *d_out;

    h_out = (float *)malloc(sizeof(float));

    cudaMalloc(&d_in, N * sizeof(__half));
    cudaMalloc(&d_out, sizeof(float));

    initialize_random_normal(d_in, N);

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

    double bytes = double(N) * sizeof(__half);
    double bandwidth = bytes / (milliseconds / 100.0 / 1e3) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    cudaMemset(d_out, 0, sizeof(float));
    launch_reduction(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("the sum is %f\n", h_out[0]);
    verify_reduction(d_in, h_out, N);


    cudaFree(d_in);
    cudaFree(d_out);

    free(h_out);

    return 0;
}
