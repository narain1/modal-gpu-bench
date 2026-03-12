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

cuda_src = r'''
extern "C" __global__ void vec_add(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}
'''

# prepare data
N = 1 << 18
a = Tensor.randn(N, device='CUDA', dtype="float32").realize()
b = Tensor.randn(N, device='CUDA', dtype="float32").realize()
out = Tensor.empty(N, device='CUDA', dtype="float32").realize()

# kernel arguments: match the CUDA signature (a, b, c, n)
kernel_args = [
    a.uop.buffer.ensure_allocated()._buf,
    b.uop.buffer.ensure_allocated()._buf,
    out.uop.buffer.ensure_allocated()._buf,
]
scalar_vals = (N,)

# launch config
threads = 128
blocks = (N + threads - 1) // threads
global_size = (blocks, 1, 1)
local_size = (threads, 1, 1)

# compile and run
et, prg = compile_and_run_cuda_nvcc(cuda_src, kernel_name="vec_add", nvcc_args=None,
                                    device_str="CUDA", kernel_args=kernel_args,
                                    scalar_vals=scalar_vals,
                                    global_size=global_size, local_size=local_size)

print(f"Elapsed {et*1e6:.2f} us")

# verify
ref = a.numpy() + b.numpy()
res = out.numpy()
import numpy as np
np.testing.assert_allclose(res, ref, rtol=1e-6, atol=1e-6)