#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

void initialize_random_normal(float *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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


template <unsigned int threads_per_block>
__global__ void reduction_kernel3(const float* inp, float* out, const size_t N) {
    const float4 *inp4 = reinterpret_cast<const float4*>(inp);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;

    // Process 4 float4s (16 floats) per iteration to hide memory latency
    size_t n_vectors = N / 4;
    size_t i = idx;
    
    // Main loop with unrolling factor of 4
    // We ensure i + 3 * stride is within bounds
    for (; i + 3 * stride < n_vectors; i += 4 * stride) {
        float4 v1 = inp4[i];
        float4 v2 = inp4[i + stride];
        float4 v3 = inp4[i + 2 * stride];
        float4 v4 = inp4[i + 3 * stride];

        sum += v1.x + v1.y + v1.z + v1.w;
        sum += v2.x + v2.y + v2.z + v2.w;
        sum += v3.x + v3.y + v3.z + v3.w;
        sum += v4.x + v4.y + v4.z + v4.w;
    }

    // Handle remaining float4 vectors
    for (; i < n_vectors; i += stride) {
        float4 val = inp4[i];
        sum += val.x + val.y + val.z + val.w;
    }

    // Handle remaining individual floats
    for (size_t j = (N & ~3) + idx; j < N; j += stride) {
        sum += inp[j];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(&out[0], sum);
    }
}

void verify_reduction(float *input, float *output, const size_t n) {
    double cpu_sum = 0.0;
    #pragma unroll
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += input[i];
    }
    printf("kernel is valid, %s\n", abs(cpu_sum - *output) < 1e-1 ? "true" : "false"); // relaxed tolerance for float
    printf("CPU Sum: %f\n", cpu_sum);
    printf("GPU Sum: %f\n", *output);
}

void launch_reduction(float *input, float *output, const size_t n) {
    constexpr int threads_per_block = 512;
    // Use fixed number of blocks to maximize occupancy while letting the grid-stride loop
    // handle the work. This increases ILP per thread and reduces atomic contention.
    int numBlocks = 2048; 
    reduction_kernel3<threads_per_block><<<numBlocks, threads_per_block>>>(input, output, n);
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

    printf("the sum is %f\n", h_out[0]);
    verify_reduction(h_in, h_out, N);


    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);

    return 0;
}
