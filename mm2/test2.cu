#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>
#include <stdint.h>

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void matmul_ptx_kernel_fixed(half* A, half* B, float* C, int M, int N, int K) {
    // Shared memory for 16x16 tiles of A and B
    __shared__ __align__(128) half smem_A[16 * 16];
    __shared__ __align__(128) half smem_B[16 * 16];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % 32;

    uint32_t smem_A_ptr = __cvta_generic_to_shared(smem_A);
    uint32_t smem_B_ptr = __cvta_generic_to_shared(smem_B);

    // --- Loading Phase ---
    // Load 16x16 A and B using 128 threads (4 bytes each)
    if (tid < 128) {
        int row = tid / 8;
        int col = (tid % 8) * 2;
        
        // A
        int global_offset_A = row * K + col;
        int smem_offset_A = (row * 16 + col) * sizeof(half);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(smem_A_ptr + smem_offset_A), "l"(&A[global_offset_A]));
        
        // B
        int global_offset_B = row * N + col;
        int smem_offset_B = (row * 16 + col) * sizeof(half);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(smem_B_ptr + smem_offset_B), "l"(&B[global_offset_B]));
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // --- Compute Phase (Warp 0) ---
    if (tid < 32) {
        uint32_t frag_A[4];
        uint32_t frag_B0[2];
        uint32_t frag_B1[2];
        float frag_C[8];
        for(int i=0; i<8; ++i) frag_C[i] = 0.0f;

        // Load A (16x16) -> row-major fragments
        // Mat 0 (TL): Lane 0..7
        // Mat 1 (TR): Lane 8..15
        // Mat 2 (BL): Lane 16..23
        // Mat 3 (BR): Lane 24..31
        int row_base = lane_id % 8;
        int mat_idx = lane_id / 8;
        int row_offset_A = (mat_idx / 2) * 8; // 0 or 8
        int col_offset_A = (mat_idx % 2) * 16; // 0 or 16 (bytes -> 8 halves)
        uint32_t addr_A = smem_A_ptr + (row_base + row_offset_A) * 32 + col_offset_A;

        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(frag_A[0]), "=r"(frag_A[1]), "=r"(frag_A[2]), "=r"(frag_A[3])
            : "r"(addr_A)
        );

        // Load B0 (Cols 0..7) -> Transposed -> col-major fragments
        // Mat 0 (Top): Lane 0..15? (Actually T0..7 & T8..15 redundant/same)
        // Mat 1 (Bot): Lane 16..23 & T24..31 redundant
        
        int row_B;
        if (lane_id < 16) row_B = lane_id % 8; // Mat 0 Rows 0..7
        else row_B = (lane_id % 8) + 8;       // Mat 1 Rows 8..15
        
        uint32_t addr_B0 = smem_B_ptr + row_B * 32; // Offset 0 (Cols 0..7 per row)

        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
            : "=r"(frag_B0[0]), "=r"(frag_B0[1])
            : "r"(addr_B0)
        );

        // Load B1 (Cols 8..15) -> Transposed
        uint32_t addr_B1 = addr_B0 + 16; // Offset 16 (Cols 8..15 per row)

        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
            : "=r"(frag_B1[0]), "=r"(frag_B1[1])
            : "r"(addr_B1)
        );

        // MMA A * B0 -> C[0..7 cols]
        asm volatile (
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(frag_C[0]), "=f"(frag_C[1]), "=f"(frag_C[2]), "=f"(frag_C[3])
            : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
            "r"(frag_B0[0]), "r"(frag_B0[1]),
            "f"(frag_C[0]), "f"(frag_C[1]), "f"(frag_C[2]), "f"(frag_C[3])
        );

        // MMA A * B1 -> C[8..15 cols]
        asm volatile (
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(frag_C[4]), "=f"(frag_C[5]), "=f"(frag_C[6]), "=f"(frag_C[7])
            : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]),
            "r"(frag_B1[0]), "r"(frag_B1[1]),
            "f"(frag_C[4]), "f"(frag_C[5]), "f"(frag_C[6]), "f"(frag_C[7])
        );

        // --- Store ---
        int group = lane_id / 4;
        int sub_col = (lane_id % 4) * 2;
        int row0 = group;
        int row1 = group + 8;
        
        // C cols 0..7 (using frag_C[0..3])
        if (row0 < M && sub_col < N) C[row0 * N + sub_col] = frag_C[0];
        if (row0 < M && sub_col+1 < N) C[row0 * N + sub_col + 1] = frag_C[1];
        if (row1 < M && sub_col < N) C[row1 * N + sub_col] = frag_C[2];
        if (row1 < M && sub_col+1 < N) C[row1 * N + sub_col + 1] = frag_C[3];
        
        // C cols 8..15 (using frag_C[4..7])
        int sub_col_2 = sub_col + 8;
        if (row0 < M && sub_col_2 < N) C[row0 * N + sub_col_2] = frag_C[4];
        if (row0 < M && sub_col_2+1 < N) C[row0 * N + sub_col_2 + 1] = frag_C[5];
        if (row1 < M && sub_col_2 < N) C[row1 * N + sub_col_2] = frag_C[6];
        if (row1 < M && sub_col_2+1 < N) C[row1 * N + sub_col_2 + 1] = frag_C[7];
    }
}

// Host code
void launch_matmul_ptx(half* d_A, half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(128, 1); 
    dim3 grid(1, 1);
    matmul_ptx_kernel_fixed<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

// CPU reference implementation
void cpu_matmul(const std::vector<half>& A, const std::vector<half>& B, 
                std::vector<float>& C, int M, int N, int K) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// Validate results
bool validate_results(const std::vector<float>& gpu_result, 
                     const std::vector<float>& cpu_result, 
                     int M, int N, float tolerance = 1e-1f) {
    bool success = true;
    int error_count = 0;
    const int max_errors_to_print = 10;
    
    for(int i = 0; i < std::min(100, M * N); i++) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        if(diff > tolerance) {
            if(error_count < max_errors_to_print) {
                int row = i / N;
                int col = i % N;
                std::cout << "Error at [" << row << "][" << col << "]: GPU=" 
                         << gpu_result[i] << ", CPU=" << cpu_result[i] 
                         << ", Diff=" << diff << std::endl;
            }
            error_count++;
            success = false;
        }
    }
    
    if(!success) {
        std::cout << "Total errors: " << error_count << " out of " << std::min(100, M * N) << " checked elements" << std::endl;
    } else {
        std::cout << "Validation PASSED! All results within tolerance." << std::endl;
    }
    return success;
}

int main() {
    const int M = 16, N = 16, K = 16;
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);
    
    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_C_gpu(M * N, 0.0f);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            h_A[i * K + j] = __float2half(static_cast<float>((i % 4) + 1));
        }
    }
    
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            h_B[i * N + j] = __float2half(static_cast<float>((j % 4) + 1) * 0.5f);
        }
    }
    
    std::cout << "Computing CPU reference..." << std::endl;
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);
    
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    std::cout << "Copying data to GPU..." << std::endl;
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    launch_matmul_ptx(d_A, d_B, d_C, M, N, K);
    
    std::cout << "Copying result back to host..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    
    validate_results(h_C_gpu, h_C_cpu, M, N);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
