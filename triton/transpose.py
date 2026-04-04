import torch
import triton.language as tl
import triton

@triton.jit
def transpose_kernel(src, dst, M, N, B: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row = B * pid_m + tl.arange(0, B)
    col = B * pid_n + tl.arange(0, B)

    offset_src = row[:, None] * N + col[None, :]
    src_mask = (row[:, None] < M) & (col[None, :] < N)
    block = tl.load(src + offset_src, mask=src_mask, other=0.0)

    offset_dst = col[:, None] * M + row[None, :]
    dst_mask = (col[:, None] < N) & (row[None, :] < M)
    tl.store(dst + offset_dst, block, mask=dst_mask)

def transpose(mat):
    assert mat.is_contiguous()
    m, n = mat.shape
    dst = torch.empty((n, m), device=mat.device, dtype=mat.dtype)
    B = 32
    grid = (triton.cdiv(m, B), triton.cdiv(n, B))
    transpose_kernel[grid](src, dst, m, n, B)
    return dst

a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
b = transpose(a)
torch.testing.assert_close(a, b.T, atol=1e-3, btol=1e-3)
