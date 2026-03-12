import numpy as np
from tinygrad import Device, Tensor
from tinygrad.runtime.support.compiler_cuda import NVCCCompiler

def compile_and_run_cuda_nvcc(cuda_src: str, kernel_name: str = None, nvcc_args: list[str] = None,
                              device_str: str = "CUDA", kernel_args: list = None,
                              scalar_vals: tuple[int, ...] | None = None,
                              global_size=(1,1,1), local_size=(1,1,1), smem: int = None):
    device = Device[device_str]
    nvcc_args = nvcc_args or []
    lib = NVCCCompiler(device.compiler.arch, nvcc_args).compile(cuda_src)
    ptx = lib.decode()
    if kernel_name is None:
        kernel_name = ptx.split(".globl\t")[1].split("\n")[0]
    prg = device.runtime(kernel_name, lib)
    if smem is not None:
        prg.smem = smem
    kernel_args = kernel_args or []
    scalar_vals = scalar_vals or ()
    et = prg(*kernel_args, global_size=global_size, local_size=local_size, vals=scalar_vals, wait=True)
    return et, prg


def bench(prg, kernel_args, scalar_vals=(), global_size=(1,1,1), local_size=(1,1,1), warmup=5, iters=50):
    # warm up to stabilize clocks and cache
    for _ in range(warmup):
      prg(*kernel_args, global_size=global_size, local_size=local_size, vals=scalar_vals, wait=True)
    Device["CUDA"].synchronize()

    times = []
    for _ in range(iters):
      et = prg(*kernel_args, global_size=global_size, local_size=local_size, vals=scalar_vals, wait=True)
      times.append(et)
    Device["CUDA"].synchronize()

    arr = np.array(times)
    return {
      "mean_us": float(arr.mean() * 1e6),
      "p50_us": float(np.percentile(arr, 50) * 1e6),
      "p90_us": float(np.percentile(arr, 90) * 1e6),
      "iters": iters,
    }

# prepare kernel source - naive matmul
cuda_src = r'''
extern "C" __global__ void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
'''

# prepare data - C = A @ B
# A: M x K, B: K x N, C: M x N
M, N, K = 1024, 1024, 1024
a = Tensor.randn(M, K, device='CUDA', dtype="float32").realize()
b = Tensor.randn(K, N, device='CUDA', dtype="float32").realize()
out = Tensor.empty(M, N, device='CUDA', dtype="float32").realize()

# kernel arguments: match the CUDA signature (A, B, C, M, N, K)
kernel_args = [
    a.uop.buffer.ensure_allocated()._buf,
    b.uop.buffer.ensure_allocated()._buf,
    out.uop.buffer.ensure_allocated()._buf,
]
scalar_vals = (M, N, K)

# launch config - 2D grid and blocks
BLOCK_SIZE = 16
blocks_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
blocks_y = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
global_size = (blocks_x, blocks_y, 1)
local_size = (BLOCK_SIZE, BLOCK_SIZE, 1)

# compile and run
et, prg = compile_and_run_cuda_nvcc(cuda_src, kernel_name="matmul_naive", nvcc_args=None,
                                    device_str="CUDA", kernel_args=kernel_args,
                                    scalar_vals=scalar_vals,
                                    global_size=global_size, local_size=local_size)

print(f"Elapsed {et*1e6:.2f} us")
print(f"TFLOPS: {(2*M*N*K) / (et*1e12):.2f}")

# benchmark loop
stats = bench(prg, kernel_args, scalar_vals=scalar_vals, global_size=global_size, local_size=local_size,
              warmup=5, iters=50)
print(f"Bench mean {stats['mean_us']:.2f} us | p50 {stats['p50_us']:.2f} us | p90 {stats['p90_us']:.2f} us")
print(f"Bench TFLOPS (mean): {(2*M*N*K) / (stats['mean_us'] * 1e6):.2f}")

# verify
ref = (a @ b).numpy()
res = out.numpy()
np.testing.assert_allclose(res, ref, rtol=1e-4, atol=1e-4)
print("✓ Results match!")
