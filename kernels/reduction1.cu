#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <time.h>

void initialize_random_normal(float *data, size_t n) {
    srand(time(0));
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}

// 2 kernel approach
__global__ void reduce_sum(float *input, float *output, size_t n) {
    extern __shared__ float sdata[];
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // loading input into shared memory
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // reduction in shared memory
    // blockdim is the number of threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void reduction_kernel1(float *input, float *output, size_t n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_int;
    cudaMalloc(&d_int, numBlocks * sizeof(float));
    reduce_sum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(input, d_int, n);
    reduce_sum<<<1, blockSize, blockSize * sizeof(float)>>>(d_int, output, numBlocks);
    cudaFree(d_int);
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

float benchmark_reduction1(float *d_in, float *d_out, size_t n, int warmup = 10, int iterations = 100) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    float *d_int;
    cudaMalloc(&d_int, numBlocks * sizeof(float));

    for (int i = 0; i < warmup; ++i) {
        reduce_sum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_in, d_int, n);
        reduce_sum<<<1, blockSize, blockSize * sizeof(float)>>>(d_int, d_out, numBlocks);
    }

    float kernel_time = 0.0f;
    float milliseconds = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduce_sum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_in, d_int, n);
        reduce_sum<<<1, blockSize, blockSize * sizeof(float)>>>(d_int, d_out, numBlocks);
        cudaEventRecord(stop);
        milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        kernel_time += milliseconds;
    }

    return kernel_time / iterations;
}

int main() {
    size_t n = 1 << 16;
    float *h_in, *h_out;
    float *d_in, *d_out;

    h_in = (float *)malloc(n * sizeof(float));
    h_out = (float *)malloc(sizeof(float));
    initialize_random_normal(h_in, n);
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    reduction_kernel1(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    verify_reduction(h_in, h_out, n);
    printf("kernel 1 valid Sum: %f\n", *h_out);

    cudaMemset(d_out, 0, sizeof(float));
    float avg_time = benchmark_reduction1(d_in, d_out, n);
    printf("Average kernel time: %f ms\n", avg_time);


    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);
    return 0;

}