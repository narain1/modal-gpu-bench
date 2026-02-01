import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
import torch
import math

print("Starting softmax operation...")
start = time.time()

@cute.kernel
def softmax_kernel(input_ptr, output_ptr, row_idx, N):
    tidx, _, _ = cute.arch.thread_idx()
    
    local_max = -cute.Float32.inf
    local_sum = 0.0
    
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((1024,), stride=(1,))
    smem_tile_max = smem.allocate_tensor(cutlass.Float32, layout=smem_layout)
    smem_tile_sum = smem.allocate_tensor(cutlass.Float32, layout=smem_layout)
    
    # First pass: compute local max and sum using online algorithm
    for i in range(tidx, N, 1024):
        x = input_ptr[row_idx, i]
        x_float = x.to(cute.Float32)
        new_max = max(x_float, local_max)
        new_sum = 0.0  # Initialize before control flow
        
        if local_max == -cute.Float32.inf:
            new_sum = cute.math.exp2((x_float - new_max) * 1.4426950408889634, fastmath=True)
        else:
            new_sum = cute.math.exp2((local_max - new_max) * 1.4426950408889634, fastmath=True) * local_sum + \
                        cute.math.exp2((x_float - new_max) * 1.4426950408889634, fastmath=True)
        
        local_max = new_max
        local_sum = new_sum
    
    # Store local results to shared memory
    smem_tile_max[tidx] = local_max
    smem_tile_sum[tidx] = local_sum
    cute.arch.sync_threads()
    
    # Reduce in shared memory
    s = 1
    while s < 1024:
        if tidx % (s * 2) == 0:
            a_max = smem_tile_max[tidx]
            b_max = smem_tile_max[tidx + s]
            a_sum = smem_tile_sum[tidx]
            b_sum = smem_tile_sum[tidx + s]
            
            new_max = max(a_max, b_max)
            new_sum = 0.0  # Initialize before control flow
            
            if a_max == -cute.Float32.inf:
                new_sum = b_sum
            elif b_max == -cute.Float32.inf:
                new_sum = a_sum
            else:
                new_sum = cute.math.exp2((a_max - new_max) * 1.4426950408889634, fastmath=True) * a_sum + \
                            cute.math.exp2((b_max - new_max) * 1.4426950408889634, fastmath=True) * b_sum
            
            smem_tile_max[tidx] = new_max
            smem_tile_sum[tidx] = new_sum
        
        cute.arch.sync_threads()
        s = s * 2
    
    max_val = smem_tile_max[0]
    sum_exp = smem_tile_sum[0]
    cute.arch.sync_threads()
    
    # Second pass: write normalized values
    for i in range(tidx, N, 1024):
        x = input_ptr[row_idx, i]
        x_float = x.to(cute.Float32)
        result = cute.math.exp2((x_float - max_val) * 1.4426950408889634, fastmath=True) / sum_exp
        output_ptr[row_idx, i] = result.to(output_ptr.element_type)

@cute.jit
def softmax_op(input_tensor: cute.Tensor, output_tensor: cute.Tensor, batch_size: cutlass.Int32, seq_len: cutlass.Int32):
    """
    JIT function that launches the softmax kernel for each row.
    """
    for row in range(batch_size):
        softmax_kernel(input_tensor, output_tensor, row, seq_len).launch(
            grid=(1, 1, 1),
            block=(1024, 1, 1)
        )

print(f"Kernel compilation took {time.time() - start:.2f}s")

# Create input and output tensors for benchmarking
tensor_start = time.time()
batch_size = 128
seq_len = 2048
a_bf16 = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.bfloat16)
output_bf16 = torch.empty_like(a_bf16)
print(f"Tensor creation took {time.time() - tensor_start:.2f}s")
print(f"Input shape: {a_bf16.shape}, dtype: {a_bf16.dtype}")

# Warmup runs
print("Running warmup...")
for _ in range(10):
    softmax_op(from_dlpack(a_bf16), from_dlpack(output_bf16), batch_size, seq_len)
torch.cuda.synchronize()

# Benchmark runs
print("Running benchmark...")
num_runs = 100
exec_start = time.time()
for _ in range(num_runs):
    softmax_op(from_dlpack(a_bf16), from_dlpack(output_bf16), batch_size, seq_len)
torch.cuda.synchronize()
exec_time = time.time() - exec_start

avg_time_ms = (exec_time / num_runs) * 1000
print(f"Average kernel execution time: {avg_time_ms:.4f}ms over {num_runs} runs")
print(f"Total benchmark time: {exec_time:.2f}s")

# Verify correctness
torch_softmax = torch.nn.functional.softmax(a_bf16.float(), dim=-1).to(torch.bfloat16)
max_diff = (output_bf16.float() - torch_softmax.float()).abs().max().item()
print(f"Max difference from PyTorch softmax: {max_diff:.6e}")

print({
    "avg_time_ms": avg_time_ms,
    "total_time_s": exec_time,
    "shape": list(a_bf16.shape),
    "max_diff": max_diff
})
