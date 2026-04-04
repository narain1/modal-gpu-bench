import triton
import triton.language as tl
import torch
import platform
import sys

DEVICE = "cuda"

# C composes of bm * n_m x bn * n_n
# (m, k) (k, n) -> (m, n)
@triton.autotune(
    configs=[
        triton.Config({'b_m': 128, 'b_k': 64,  'b_n': 256, 'group': 8}, num_stages=3, num_warps=8),
        triton.Config({'b_m': 128, 'b_k': 32,  'b_n': 256, 'group': 8}, num_stages=4, num_warps=4),
        triton.Config({'b_m': 64,  'b_k': 32,  'b_n': 256, 'group': 4}, num_stages=4, num_warps=4),
        triton.Config({'b_m': 128, 'b_k': 64,  'b_n': 128, 'group': 8}, num_stages=4, num_warps=4),
        triton.Config({'b_m': 64,  'b_k': 64,  'b_n': 128, 'group': 8}, num_stages=4, num_warps=4),
        triton.Config({'b_m': 128, 'b_k': 32,  'b_n': 128, 'group': 8}, num_stages=4, num_warps=4),
    ],
    key=['m', 'n', 'k'],
)
@triton.jit
def matrix_multiply_triton(
    a, b, c,
    m, n, k,
    b_m: tl.constexpr,
    b_k: tl.constexpr,
    b_n: tl.constexpr,
    group: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_blocks_m = tl.cdiv(m, b_m)
    n_blocks_n = tl.cdiv(n, b_n)

    # L2 swizzling: cluster blocks into (group x n_blocks_n) slabs so adjacent
    # SMs share the same A rows in L2, cutting redundant HBM traffic
    num_pid_in_group = group * n_blocks_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group
    group_size_m = tl.minimum(n_blocks_m - first_pid_m, group)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row_start = pid_m * b_m
    col_start = pid_n * b_n

    acc = tl.zeros((b_m, b_n), dtype=tl.float32)
    for k_idx in range(0, k, b_k):
        offset_a = (tl.arange(0, b_m)[:, None] + row_start) * k + (tl.arange(0, b_k)[None, :] + k_idx)
        offset_b = (k_idx + tl.arange(0, b_k))[:, None] * n + (col_start + tl.arange(0, b_n)[None, :])

        mask_a = (row_start + tl.arange(0, b_m)[:, None] < m) & (k_idx + tl.arange(0, b_k)[None, :] < k)
        mask_b = (k_idx + tl.arange(0, b_k)[:, None] < k) & (col_start + tl.arange(0, b_n)[None, :] < n)

        a_tile = tl.load(a + offset_a, mask=mask_a, other=0.0)
        b_tile = tl.load(b + offset_b, mask=mask_b, other=0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    offset_c = (row_start + tl.arange(0, b_m))[:, None] * n + (col_start + tl.arange(0, b_n))[None, :]
    mask_c = (row_start + tl.arange(0, b_m)[:, None] < m) & (col_start + tl.arange(0, b_n)[None, :] < n)
    tl.store(c + offset_c, acc.to(tl.float16), mask=mask_c)


def mm_triton(a, b):
    m, k = a.shape
    k, n = b.shape

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(m, meta['b_m']) * triton.cdiv(n, meta['b_n']),)
    matrix_multiply_triton[grid](a, b, c, m, n, k)
    return c


# Correctness check
a = torch.randn((2048, 2048), device=DEVICE, dtype=torch.float16)
b = torch.randn((2048, 2048), device=DEVICE, dtype=torch.float16)
c_ref = (a @ b).half()
c_out = mm_triton(a, b)
torch.testing.assert_close(c_ref, c_out, atol=1e-3, rtol=1e-3)
print("kernel passed")

print("python:", sys.version)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))


def tflops(size, ms):
    return 2 * size ** 3 * 1e-12 / (ms * 1e-3)


print(f"\n{'size':>6}  {'Triton (TFLOPS)':>18}  {'Torch (TFLOPS)':>16}")
for size in [512, 1024, 2048, 4096, 8192]:
    a = torch.randn((size, size), device=DEVICE, dtype=torch.float16)
    b = torch.randn((size, size), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    triton_ms, _, _ = triton.testing.do_bench(lambda: mm_triton(a, b), quantiles=quantiles)
    torch_ms, _, _ = triton.testing.do_bench(lambda: a @ b, quantiles=quantiles)

    print(f"{size:>6}  {tflops(size, triton_ms):>18.2f}  {tflops(size, torch_ms):>16.2f}")
