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

int main() {
    size_t N = 1 << 16;
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

    reduction_kernel2<threads_per_block><<<block_dim, threads_per_block>>>(d_in, d_int, N);
    reduction_kernel2<threads_per_block><<<1, threads_per_block>>>(d_int, d_out, block_dim);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("the sum is %f\n", h_out[0]);
    verify_reduction(h_in, h_out, N);


    cudaFree(d_in);
    cudaFree(d_int);
    cudaFree(d_out);

    free(h_in);
    free(h_out);

    return 0;
}
