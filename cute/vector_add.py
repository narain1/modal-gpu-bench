import modal
import statistics

app = modal.App("base")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "gcc")
    .pip_install(
        "torch",
        "triton",
        "nvidia-cutlass-dsl==4.3.2",  # provides cutlass & cutlass.cute
        "numpy",
    )
)

@app.function(image=image, gpu="T4", timeout=600)
def cutedsl_benchmark(sizes: list[int]):
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import statistics

    # Constants
    VECTOR_WIDTH = 8  # float32x8 vector load
    THREADS_PER_BLOCK = 256


    @cute.kernel
    def vectorized_add_1d_kernel(
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        thread_idx = bidx * bdim + tidx
        
        num_vectors = cute.size(gA, mode=[1])
        
        if thread_idx < num_vectors:
            a_val = gA[(None, thread_idx)].load()
            b_val = gB[(None, thread_idx)].load()
            
            gC[(None, thread_idx)] = a_val + b_val


    @cute.jit
    def solution(d_input1: cute.Tensor, d_input2: cute.Tensor, d_output: cute.Tensor, n: cute.Int32):
        threads_per_block = 256
        
        gA = cute.zipped_divide(d_input1, (32,))
        gB = cute.zipped_divide(d_input2, (32,))
        gC = cute.zipped_divide(d_output, (32,))
        
        # Calculate number of vectors (n // 4)
        # Use mode=[1] instead of mode=1
        num_vectors = cute.size(gC, mode=[1])
        
        # Launch kernel
        num_blocks = (num_vectors + threads_per_block - 1) // threads_per_block
        vectorized_add_1d_kernel(gA, gB, gC).launch(
            grid=(num_blocks, 1, 1),
            block=(threads_per_block, 1, 1),
        )

    def cutedsl_add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
        assert x.is_cuda and x.dtype == torch.float32
        assert x.numel() % VECTOR_WIDTH == 0, f"Size must be divisible by {VECTOR_WIDTH}"

        tx = from_dlpack(x)
        ty = from_dlpack(y)
        tout = from_dlpack(output)

        gA = cute.zipped_divide(tx, (VECTOR_WIDTH,))
        gB = cute.zipped_divide(ty, (VECTOR_WIDTH,))
        gC = cute.zipped_divide(tout, (VECTOR_WIDTH,))

        num_vectors = cute.size(gC, mode=[1])
        num_blocks = (num_vectors + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        vectorized_add_1d_kernel(gA, gB, gC).launch(
            grid=(num_blocks, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )

    def benchmark_size(size: int, warmup=10, iterations=100):
        # Align to VECTOR_WIDTH
        aligned_size = ((size + VECTOR_WIDTH - 1) // VECTOR_WIDTH) * VECTOR_WIDTH
        if aligned_size != size:
            pass  # silent alignment (or log if needed)

        x = torch.randn(aligned_size, device="cuda", dtype=torch.float32)
        y = torch.randn(aligned_size, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        # Warmup
        for _ in range(warmup):
            cutedsl_add(x, y, out)
        torch.cuda.synchronize()

        # Timed runs
        times_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            cutedsl_add(x, y, out)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))  # in ms

        avg_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

        # Compute metrics
        bytes_moved = aligned_size * 4 * 3  # 3 tensors × 4 bytes each
        bandwidth_gbs = (bytes_moved / avg_ms) / 1e6  # bytes/ms → GB/s
        # Add: 1 flop per element
        gflops = (aligned_size / avg_ms) / 1e6  # elements/ms → GFLOP/s

        # Correctness check (on a subset to save time if huge)
        check_size = min(1024, aligned_size)
        ref = x[:check_size] + y[:check_size]
        actual = out[:check_size]
        is_correct = torch.allclose(ref, actual, rtol=1e-5, atol=1e-7)

        return {
            "original_size": size,
            "aligned_size": aligned_size,
            "avg_time_ms": avg_ms,
            "std_time_ms": std_ms,
            "bandwidth_gbs": bandwidth_gbs,
            "gflops": gflops,
            "is_correct": is_correct,
        }

    results = {}
    for size in sizes:
        try:
            res = benchmark_size(size)
            results[size] = res
        except Exception as e:
            results[size] = {"error": str(e)}

    # GPU info
    try:
        gpu_name = torch.cuda.get_device_name()
    except Exception:
        gpu_name = "unknown"

    return {
        "gpu_name": gpu_name,
        "vector_width": VECTOR_WIDTH,
        "results": results,
    }


@app.local_entrypoint()
def main():
    sizes = [2**15, 2**20, 2**25, 2**30]
    print("Launching CuTeDSL benchmark on Modal GPU...")
    result = cutedsl_benchmark.remote(sizes)

    print(f"\nGPU: {result['gpu_name']}")
    print(f"Vector width: {result['vector_width']}")
    print("=" * 80)
    print(f"{'Size':>12} | {'Aligned':>10} | {'Avg ms':>8} | {'BW (GB/s)':>10} | {'GFLOP/s':>9} | {'Correct'}")
    print("-" * 80)
    for size, res in result["results"].items():
        if "error" in res:
            print(f"{size:>12} | ✗ ERROR: {res['error']}")
        else:
            print(
                f"{size:>12,} | {res['aligned_size']:>10,} | "
                f"{res['avg_time_ms']:>8.3f} | {res['bandwidth_gbs']:>10.2f} | "
                f"{res['gflops']:>9.2f} | {'✓' if res['is_correct'] else '✗'}"
            )