import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
import torch
import math
import os

# Read GPU and timeout from environment variables (set by run_rms_norm.py)
gpu = os.environ.get('GPU', 'H100')
timeout = int(os.environ.get('TIMEOUT', '600'))

print(f"Starting RMS Norm operation on {gpu}...")
start = time.time()

@cute.kernel
def rms_norm_kernel(input_ptr, output_ptr, gamma_ptr, row_idx, N, eps):
    """
    RMS Normalization kernel.
    RMS = sqrt(mean(x^2) + eps)
    output = (x / RMS) * gamma
    """
    tidx, _, _ = cute.arch.thread_idx()
    
    local_sum_sq = 0.0
    
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((1024,), stride=(1,))
    smem_tile = smem.allocate_tensor(cutlass.Float32, layout=smem_layout)
    
    # First pass: compute sum of squares
    for i in range(tidx, N, 1024):
        x = input_ptr[row_idx, i]
        x_float = x.to(cute.Float32)
        local_sum_sq = local_sum_sq + x_float * x_float
    
    # Store local results to shared memory
    smem_tile[tidx] = local_sum_sq
    cute.arch.sync_threads()
    
    # Reduce in shared memory
    s = 1
    while s < 1024:
        if tidx % (s * 2) == 0:
            smem_tile[tidx] = smem_tile[tidx] + smem_tile[tidx + s]
        cute.arch.sync_threads()
        s = s * 2
    
    # Compute RMS
    sum_sq = smem_tile[0]
    mean_sq = sum_sq / cute.Float32(N)
    rms = cute.math.sqrt(mean_sq + eps, fastmath=True)
    cute.arch.sync_threads()
    
    # Second pass: normalize and scale
    for i in range(tidx, N, 1024):
        x = input_ptr[row_idx, i]
        x_float = x.to(cute.Float32)
        gamma_val = gamma_ptr[i]
        gamma_float = gamma_val.to(cute.Float32)
        result = (x_float / rms) * gamma_float
        output_ptr[row_idx, i] = result.to(output_ptr.element_type)

@cute.jit
def rms_norm_op(input_tensor: cute.Tensor, output_tensor: cute.Tensor, 
                gamma_tensor: cute.Tensor, batch_size: cutlass.Int32, 
                hidden_size: cutlass.Int32, epsilon: cutlass.Float32):
    """
    JIT function that launches the RMS norm kernel for each row.
    """
    for row in range(batch_size):
        rms_norm_kernel(input_tensor, output_tensor, gamma_tensor, 
                    row, hidden_size, epsilon).launch(
            grid=(1, 1, 1),
            block=(1024, 1, 1)
        )

print(f"Kernel compilation took {time.time() - start:.2f}s")

# Create input and output tensors for benchmarking
tensor_start = time.time()
batch_size = 128
hidden_size = 2048
eps = 1e-5

a_bf16 = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.bfloat16)
gamma_bf16 = torch.ones(hidden_size, device='cuda', dtype=torch.bfloat16)
output_bf16 = torch.empty_like(a_bf16)
print(f"Tensor creation took {time.time() - tensor_start:.2f}s")
print(f"Input shape: {a_bf16.shape}, dtype: {a_bf16.dtype}")

# Warmup runs
print("Running warmup...")
for _ in range(10):
    rms_norm_op(from_dlpack(a_bf16), from_dlpack(output_bf16), 
                from_dlpack(gamma_bf16), batch_size, hidden_size, eps)
torch.cuda.synchronize()

# Benchmark runs
print("Running benchmark...")
num_runs = 100
exec_start = time.time()
for _ in range(num_runs):
    rms_norm_op(from_dlpack(a_bf16), from_dlpack(output_bf16), 
                from_dlpack(gamma_bf16), batch_size, hidden_size, eps)
torch.cuda.synchronize()
exec_time = time.time() - exec_start

avg_time_ms = (exec_time / num_runs) * 1000
print(f"Average kernel execution time: {avg_time_ms:.4f}ms over {num_runs} runs")
print(f"Total benchmark time: {exec_time:.2f}s")

# Verify correctness against PyTorch implementation
def torch_rms_norm(x, gamma, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return x_normalized * gamma

torch_output = torch_rms_norm(a_bf16.float(), gamma_bf16.float(), eps).to(torch.bfloat16)
max_diff = (output_bf16.float() - torch_output.float()).abs().max().item()
mean_diff = (output_bf16.float() - torch_output.float()).abs().mean().item()
print(f"Max difference from PyTorch RMS norm: {max_diff:.6e}")
print(f"Mean difference from PyTorch RMS norm: {mean_diff:.6e}")

print({
    "gpu": gpu,
    "timeout": timeout,
    "avg_time_ms": avg_time_ms,
    "total_time_s": exec_time,
    "shape": list(a_bf16.shape),
    "max_diff": max_diff,
    "mean_diff": mean_diff
})

