#include <cuda_runtime.h>
#include <random>
#include <iostream>


void initialize_random_normal(float *data, size_t n) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = distribution(generator);
    }
}


__global__ void matrix_multiply_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}


void mm_cpu(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}



int main() {
    int m = 2048, n = 2048, k = 2048;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    A = (float*)malloc(size_A);
    B = (float*)malloc(size_B);
    C = (float*)malloc(size_C);
    initialize_random_normal(A, m * k);
    initialize_random_normal(B, k * n);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    for (int i = 0; i < 10; ++i) {
        matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Average time for matrix multiplication: %f ms\n", milliseconds / 100.0);
    double tflops = static_cast<double>(2) * m * n * k / (milliseconds / 100.0 / 1e3) / 1e12;
    printf("TFLOPS: %f\n", tflops);
    double bytes = double(m * k + k * n + m * n) * sizeof(float);
    double bandwidth = bytes / (milliseconds / 100.0 / 1e3) / 1e9;
    printf("Bandwidth: %f GB/s\n", bandwidth);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    float *C_cpu = (float*)malloc(size_C);
    mm_cpu(A, B, C_cpu, m, n, k);
    // verify results
    bool correct = true;
    for (int i = 0; i < m * n; ++i) {
        if (fabs(C[i] - C_cpu[i]) > 1e-3)
        {
            correct = false;
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, C[i], C_cpu[i]);
            break;
        }
    }
    if (correct) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect!\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
}
