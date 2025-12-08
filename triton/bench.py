import modal

app = modal.App("triton-add-benchmark")

image = (
    modal.Image. debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "triton",
        "numpy",
    )
)


def run_triton_benchmark(gpu_name: str, sizes: list[int]):
    """Run Triton add kernel benchmark - shared implementation"""
    import torch
    import triton
    import triton.language as tl
    import statistics
    # Inline TritonBenchmark to avoid ModuleNotFoundError when running remotely
    class TritonBenchmark:
        def get_kernel_code(self) -> str:
            # return kernel and helper function definitions as source
            return '''
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
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
                elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
                times.append(elapsed_time)

            avg_time_ms = statistics.mean(times)
            std_time_ms = statistics.stdev(times) if len(times) > 1 else 0
            min_time_ms = min(times)
            max_time_ms = max(times)

            bytes_transferred = size * 4 * 3  # 4 bytes per float32
            bandwidth_gbs = (bytes_transferred / (avg_time_ms / 1000)) / 1e9

            flops = size  # Total floating point operations
            gflops = (flops / (avg_time_ms / 1000)) / 1e9  # GFLOPS

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

    benchmark = TritonBenchmark()
    exec(benchmark.get_kernel_code(), globals())
    
    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }
    
    print(f"\n{'='*60}")
    print(f"Benchmarking on {gpu_name}")
    print(f"Device: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"{'='*60}\n")
    
    results = []
    for size in sizes:
        print(f"Benchmarking size: {size:,} elements...")
        result = benchmark.benchmark_add_kernel(size)
        results.append(result)
        
        print(f"  Average Time: {result['avg_time_ms']:.4f} ms")
        print(f"  Bandwidth: {result['bandwidth_gbs']:.2f} GB/s")
        print(f"  GFLOPS: {result['gflops']:.2f}")
        print(f"  Correct: {result['is_correct']}")
        print()
    
    return {
        "gpu_name": gpu_name,
        "gpu_info": gpu_info,
        "results": results,
    }


@app.function(image=image, gpu="T4", timeout=600)
def benchmark_t4(sizes: list[int]):
    return run_triton_benchmark("T4", sizes)


@app.function(image=image, gpu="L4", timeout=600)
def benchmark_l4(sizes: list[int]):
    return run_triton_benchmark("L4", sizes)


@app.function(image=image, gpu="A10G", timeout=600)
def benchmark_a10g(sizes: list[int]):
    return run_triton_benchmark("A10G", sizes)


@app.function(image=image, gpu="A100-80GB", timeout=600)
def benchmark_a100_80gb(sizes: list[int]):
    return run_triton_benchmark("A100-80GB", sizes)


@app.function(image=image, gpu="H100", timeout=600)
def benchmark_h100(sizes: list[int]):
    return run_triton_benchmark("H100", sizes)


## H100/H100!
# H200
# B200
# Map GPU names to their functions
GPU_BENCHMARK_MAP = {
    "T4": benchmark_t4,
    "L4": benchmark_l4,
    "A10G": benchmark_a10g,
    "A100-80GB": benchmark_a100_80gb,
    "H100": benchmark_h100,
}


@app.local_entrypoint()
def main(
    gpus: str = "T4,L4,A10G,A100-80GB,H100",
    sizes: str = "1024,16384,262144,1048576,16777216,67108864"
):
    """
    Main entry point to benchmark across GPUs
    
    Args:
        gpus: Comma-separated list of GPU types to benchmark
        sizes: Comma-separated list of problem sizes
    """
    # Parse input arguments
    gpu_list = [g.strip() for g in gpus.split(",")]
    size_list = [int(s. strip()) for s in sizes.split(",")]
    
    print("\n" + "="*60)
    print("Triton Add Kernel Benchmark - Multi-GPU")
    print("="*60)
    print(f"GPUs to test: {gpu_list}")
    print(f"Problem sizes: {[f'{s:,}' for s in size_list]}")
    print()
    
    all_results = []
    
    # Benchmark on each GPU type
    for gpu_name in gpu_list:
        if gpu_name not in GPU_BENCHMARK_MAP:
            print(f"⚠ Skipping unknown GPU: {gpu_name}")
            continue
            
        try:
            print(f"\n{'='*60}")
            print(f"Launching benchmark on {gpu_name}...")
            print(f"{'='*60}")
            
            # Get the benchmark function for this GPU
            benchmark_func = GPU_BENCHMARK_MAP[gpu_name]
            
            # Run the benchmark
            result = benchmark_func.remote(size_list)
            all_results.append(result)
            
            print(f"✓ Completed benchmark on {gpu_name}")
            
        except Exception as e:
            print(f"✗ Error benchmarking on {gpu_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if not all_results:
        print("No successful benchmarks completed.")
        return
    
    for gpu_result in all_results:
        print(f"\n{gpu_result['gpu_name']} - {gpu_result['gpu_info']['name']}")
        print(f"Memory: {gpu_result['gpu_info']['total_memory_gb']:.2f} GB")
        print(f"Compute Capability: {gpu_result['gpu_info']['compute_capability']}")
        print("-" * 80)
        print(f"{'Size':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20} {'GFLOPS':<15}")
        print("-" * 80)
        
        for result in gpu_result['results']:
            print(f"{result['size']:<15,} {result['avg_time_ms']:<15.4f} {result['bandwidth_gbs']:<20.2f} {result['gflops']:<15.2f}")
    
    # Print bandwidth comparison table
    print("\n" + "="*60)
    print("BANDWIDTH COMPARISON (GB/s)")
    print("="*60)
    
    # Get unique sizes
    sizes_list = all_results[0]['results']
    
    # Print header
    header = f"{'Size':<15}"
    for gpu_result in all_results:
        header += f" {gpu_result['gpu_name']:<12}"
    print(header)
    print("-" * (15 + 12 * len(all_results)))
    
    # Print bandwidth for each size
    for i, size_info in enumerate(sizes_list):
        row = f"{size_info['size']:<15,}"
        for gpu_result in all_results:
            bandwidth = gpu_result['results'][i]['bandwidth_gbs']
            row += f" {bandwidth:<12.2f}"
        print(row)
    
    # Print GFLOPS comparison table
    print("\n" + "="*60)
    print("GFLOPS COMPARISON")
    print("="*60)
    
    # Print header
    header = f"{'Size':<15}"
    for gpu_result in all_results:
        header += f" {gpu_result['gpu_name']:<12}"
    print(header)
    print("-" * (15 + 12 * len(all_results)))
    
    # Print GFLOPS for each size
    for i, size_info in enumerate(sizes_list):
        row = f"{size_info['size']:<15,}"
        for gpu_result in all_results:
            gflops = gpu_result['results'][i]['gflops']
            row += f" {gflops:<12.2f}"
        print(row)
    
    print("\n" + "="*60)