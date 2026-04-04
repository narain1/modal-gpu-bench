"""
Tensor Core GEMM: A Beginner's Guide to GPU Matrix Multiplication
===================================================================

This tutorial walks you through implementing a high-performance matrix multiplication
using NVIDIA Tensor Cores. We'll build up from simple concepts to a working kernel.

Prerequisites:
- Basic understanding of CUDA programming
- Familiarity with matrix operations
- Access to a GPU with Tensor Cores (sm80+)

Written in the spirit of Jeremy Howard's educational approach:
executable code first, then deep explanations.
"""

import torch
from torch.utils.cpp_extension import load_inline
import time
from pathlib import Path

# ============================================================================
# PART 1: UNDERSTANDING THE PROBLEM
# ============================================================================
"""
What is GEMM?
-------------
GEMM = GEneral Matrix Multiply

Given matrices A (M×K) and B (K×N), compute C = A × B (M×N)

For our tutorial, we'll use:
- M = N = K = 4096 (multiples of 128 and 32 for optimal Tensor Core usage)
- Element type: half-precision floating point (fp16)
- Output type: single-precision floating point (fp32) for accumulation

Why Tensor Cores?
-----------------
Traditional CUDA cores: 1 operation per cycle per core
Tensor Cores: 4×4×4 matrix multiply-accumulate per cycle

mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
- Input: two 16×8 fp16 matrices
- Output: one 16×8 fp32 matrix
- Does: D = A × B + C (multiply-accumulate)
"""

# ============================================================================
# PART 2: THE KERNEL - STEP BY STEP
# ============================================================================

SRC = Path('mm2/mm_src.cu').read_text()

# ============================================================================
# PART 3: COMPILING AND RUNNING
# ============================================================================

print("=" * 70)
print("COMPILING THE KERNEL")
print("=" * 70)

mod = load_inline(
    name="mm_tc_3_4",
    cpp_sources="",
    cuda_sources=SRC,
    functions=None,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90"],
    with_cuda=True,
    verbose=True,
    is_python_module=False,
)

# ============================================================================
# PART 4: TESTING
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING CORRECTNESS TEST")
print("=" * 70)

M, K, N = 4096, 4096, 4096

# Create random matrices
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

# Bcol = B transposed (stored as [N, K] for column-major access)
Bcol = B.t().contiguous()

# Run our kernel
C = torch.ops.lib.mm_3_4(A, Bcol)

# Compare with PyTorch's reference implementation
# Note: A @ Bcol.t() = A @ B since Bcol = B.t()
expected = torch.matmul(A, B)

torch.testing.assert_close(C.to(torch.float16), expected)

print("✓ Correctness test PASSED!")

# ============================================================================
# PART 5: BENCHMARKING
# ============================================================================

print("\n" + "=" * 70)
print("BENCHMARKING")
print("=" * 70)

# Warmup runs
print("Warming up...")
for _ in range(10):
    C = torch.ops.lib.mm_3_4(A, Bcol)

# Synchronize before timing
torch.cuda.synchronize()

# Timed runs
num_runs = 100
start = time.perf_counter()
for _ in range(num_runs):
    C = torch.ops.lib.mm_3_4(A, Bcol)
torch.cuda.synchronize()
end = time.perf_counter()

avg_ms = (end - start) / num_runs * 1000

print(f"\nMatrix sizes: {M}×{K} × {K}×{N}")
print(f"Average time per matmul: {avg_ms:.6f} ms")

# Calculate TFLOPS (2*M*K*N floating point ops for GEMM)
tflops = 2 * M * N * K / (avg_ms * 1e-3) / 1e12
print(f"TFLOPS: {tflops:.6f}")

# Bandwidth calculation
bytes_transferred = (M * K + K * N + M * N) * 2  # fp16 = 2 bytes
bandwidth_gbps = bytes_transferred / (avg_ms * 1e-3) / 1e9
print(f"Bandwidth: {bandwidth_gbps:.6f} GB/s")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. TENSOR CORES ARE FAST: They perform 4×4×4 = 64 FLOPs per instruction
   vs. 1 FLOP per FMA instruction on CUDA cores.

2. MEMORY HIDING: The async copy pipeline keeps the Tensor Cores fed
   with data while they compute.

3. SHARED MEMORY: Used as a fast buffer between global memory and
   Tensor Core registers. Bank conflicts are avoided via swizzling.

4. REGISTER PRESSURE: We carefully balance register usage. Too many
   registers = fewer warps per SM = lower occupancy.

5. PROPER ALIGNMENT: Tensor Cores require specific memory layouts.
   The ldmatrix instruction enforces these constraints.

Try experimenting with:
- Different matrix sizes (must be multiples of 128 for N, 128 for M, 32 for K)
- Different N_STAGES values
- Different block sizes
""")
