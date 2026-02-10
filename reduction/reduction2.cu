#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <cmath>

__inline__ void initialize_random_normal(float *data, size_t n) {
    srand(time(0));
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}

template <unsigned int threads_per_block>
__global__ void reduction_kernel2(const float *inp, float *out, size_t N) {
    __shared__ float smem[threads_per_block];
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    // loading data into shared memory
    float sum = 0.0f;
    for (size_t i=gid; i<N; i+=stride) sum += inp[i];

    smem[tid] = sum;
    __syncthreads();

    for (int active = threads_per_block >> 1; active > 32; active >>= 1) {
        if (tid < active) {
            smem[tid] += smem[active + tid];
        }
        __syncthreads();
    }
  
    if (tid < 32) {
      volatile float *volatile_sum = smem;
      #pragma unroll
      for (int offset=32; offset > 0; offset >>=1) {
          volatile_sum[tid] += volatile_sum[tid + offset];
      }
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[tid];
    }
}

void verify_reduction(float *input, float *output, size_t n) {
    float cpu_sum = 0.0f;
    #pragma unroll
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += input[i];
    }
    printf("kernel is valid, %s\n", abs(cpu_sum - *output) < 1e-3 ? "true" : "false");
    printf("CPU Sum: %f\n", cpu_sum);
    printf("GPU Sum: %f\n", *output);
}


void launch_reduction(float *input, float *output, const size_t n) {
    constexpr int threads_per_block = 512;
    size_t numBlocks = (n + threads_per_block - 1) / threads_per_block;
    float *d_int;
    cudaMalloc(&d_int, numBlocks * sizeof(float));
    reduction_kernel2<threads_per_block><<<numBlocks, threads_per_block>>>(input, d_int, n);
    reduction_kernel2<threads_per_block><<<1, threads_per_block>>>(d_int, output, numBlocks);
    cudaFree(d_int);
}


int main() {
    size_t N = 1 << 24;
    float *h_in, *h_out;
    float *d_in, *d_int, *d_out;

    h_in = (float *)malloc(N * sizeof(float));
    h_out = (float *)malloc(sizeof(float));
    initialize_random_normal(h_in, N);
    constexpr unsigned int threads_per_block = 256;
    int block_dim = (N + threads_per_block - 1) / threads_per_block;

    cudaMalloc(&d_int, block_dim * sizeof(float));
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

    cudaMemset(d_out, 0, sizeof(float));
    launch_reduction(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("the sum is %f\n", h_out[0]);
    verify_reduction(h_in, h_out, N);

    double tflops = static_cast<double>(N) * 2.0 / (
        milliseconds / 100.0 / 1e3) / 1e12;
    printf("TFLOPS: %f\n", tflops);

    double bytes = double(N) * sizeof(float);
    double bandwidth = bytes / (milliseconds / 100.0 / 1e3) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    cudaFree(d_in);
    cudaFree(d_int);
    cudaFree(d_out);

    free(h_in);
    free(h_out);

    return 0;
}
