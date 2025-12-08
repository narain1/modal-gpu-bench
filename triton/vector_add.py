import torch
import triton
import triton.language as tl
import statistics

class TritonBenchmark:
    def get_kernel_code(self):
        return '''
@triton.jit
def add_kernel(
    x_ptr,  
    y_ptr,  
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,  
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
'''

    def benchmark_add_kernel(self, size: int, num_iterations: int = 100, warmup: int = 10):
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        
        for _ in range(warmup):
            _ = triton_add(x, y)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = triton_add(x, y)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  
            times.append(elapsed_time)
        
        avg_time_ms = statistics.mean(times)
        std_time_ms = statistics.stdev(times) if len(times) > 1 else 0
        min_time_ms = min(times)
        max_time_ms = max(times)
        
        bytes_transferred = size * 4 * 3  
        bandwidth_gbs = (bytes_transferred / (avg_time_ms / 1000)) / 1e9
        
        flops = size  
        gflops = (flops / (avg_time_ms / 1000)) / 1e9 
        
        expected = x + y
        is_correct = torch.allclose(output, expected, rtol=1e-5)
        
        return {
            "size": size,
            "avg_time_ms": avg_time_ms,
            "std_time_ms": std_time_ms,
            "min_time_ms": min_time_ms,
            "max_time_ms": max_time_ms,
            "bandwidth_gbs": bandwidth_gbs,
            "gflops": gflops,
            "is_correct": is_correct,
            "num_iterations": num_iterations,
        }