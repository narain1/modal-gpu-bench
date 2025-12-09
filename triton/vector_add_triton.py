import modal

app = modal.App("base")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "gcc")  
    .pip_install(
        "torch",
        "triton",
        "numpy",
        "nvidia-cutlass-dsl==4.3.2",  
    )
)

@app.function(image=image, gpu="T4", timeout=600)
def kernel(sizes: list[int]):
    import torch
    import triton
    import triton.language as tl
    import time
    from triton.testing import do_bench

    @triton.jit
    def add_kernel(
        x_ptr, y_ptr, output_ptr, N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        offs = offsets.to(tl.int32)
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        result = x + y
        tl.store(output_ptr + offs, result, mask=mask)

    def benchmark_add(size: int):
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

        def call_kernel():
            add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)

        ms = do_bench(
            call_kernel,
            warmup=25,    
            rep=100,      
            return_mode="median", 
        )
        return ms

    results = {}
    for size in sizes:
        elapsed_time = benchmark_add(size)
        results[size] = elapsed_time * 1e-3

    try:
        gpu_name = triton.testing.get_gpu_name()
        gpu_info = triton.testing.get_gpu_info()
    except Exception:
        gpu_name = "unknown"
        gpu_info = {}

    return {
        "gpu_name": gpu_name,
        "gpu_info": gpu_info,
        "results": results
    }

@app.local_entrypoint()
def main():
    sizes = [2**10, 2**15, 2**20, 2**25, 2**30]
    result = kernel.remote(sizes)
    print("GPU Name:", result["gpu_name"])
    print("GPU Info:", result["gpu_info"])
    for size, time_taken in result["results"].items():
        print(f"Size: {size}, Time taken: {time_taken:.6f} seconds")