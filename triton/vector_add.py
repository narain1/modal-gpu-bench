import torch, triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x, y, z, n_elements, B: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * B + tl.arange(0, B)
    mask = offset < n_elements
    xb = tl.load(x + offset, mask=mask, other=0.0)
    yb = tl.load(y + offset, mask=mask, other=0.0)
    zb = xb + yb
    tl.store(z + offset, zb)

def vector_add(x, y):
    n = x.shape[0]
    assert n == y.shape[0] and len(x.shape) == len(y.shape)
    z = torch.empty(n, device=x.device)
    B = 256
    grid = (triton.cdiv(n, B),)
    vector_add_kernel[grid](x, y, z, n, B)
    torch.testing.assert_close(x + y, z, atol=1e-3, rtol=1e-3)
    print("vector add passed")

n = 1 << 12
vector_add(torch.randn(n, device="cuda"), torch.randn(n, device="cuda"))
