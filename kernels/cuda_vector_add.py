import torch
from torch.utils.cpp_extension import load_inline
import time

src = r"""
#include <torch/extension.h>
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
torch::Tensor launch(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch, "vec_add"); }
"""

ext = load_inline(
    name="vec_add_ext",
    cpp_sources="",
    cuda_sources=src,
    functions=None,
    extra_cuda_cflags=["-lineinfo"],
    verbose=False,
)


def benchmark(fn, *args, warmup=10, iters=100):
    """Benchmark a function with warmup and timing.
    
    Args:
        fn: Function to benchmark
        *args: Arguments to pass to the function
        warmup: Number of warmup iterations
        iters: Number of timed iterations
        
    Returns:
        dict with timing statistics in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    
    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start_event.record()
        _ = fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    times = torch.tensor(times)
    return {
        "mean_ms": times.mean().item(),
        "median_ms": times.median().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
        "std_ms": times.std().item(),
    }


if __name__ == "__main__":
    a = torch.randn(1 << 20, device="cuda")
    b = torch.randn_like(a)
    out = ext.launch(a, b)
    torch.testing.assert_close(out, a + b)
    print("Correctness check: ok")
    
    # Benchmark custom CUDA kernel
    print(f"\nBenchmarking custom CUDA kernel (n={a.numel():,})...")
    cuda_stats = benchmark(ext.launch, a, b)
    print(f"  Mean: {cuda_stats['mean_ms']:.4f} ms")
    print(f"  Median: {cuda_stats['median_ms']:.4f} ms")
    print(f"  Min: {cuda_stats['min_ms']:.4f} ms")
    print(f"  Max: {cuda_stats['max_ms']:.4f} ms")
    print(f"  Std: {cuda_stats['std_ms']:.4f} ms")
    
    # Benchmark PyTorch native
    print(f"\nBenchmarking PyTorch native (n={a.numel():,})...")
    torch_stats = benchmark(lambda a, b: a + b, a, b)
    print(f"  Mean: {torch_stats['mean_ms']:.4f} ms")
    print(f"  Median: {torch_stats['median_ms']:.4f} ms")
    print(f"  Min: {torch_stats['min_ms']:.4f} ms")
    print(f"  Max: {torch_stats['max_ms']:.4f} ms")
    print(f"  Std: {torch_stats['std_ms']:.4f} ms")
    
    speedup = torch_stats['mean_ms'] / cuda_stats['mean_ms']
    print(f"\nSpeedup: {speedup:.2f}x {'(CUDA faster)' if speedup > 1 else '(PyTorch faster)'}")
