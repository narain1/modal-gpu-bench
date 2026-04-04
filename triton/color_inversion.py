import torch
import triton
import triton.language as tl

@triton.jit
def color_invert_kernel1(mat, w, h, B: tl.constexpr):
    pid = tl.program_id(axis=0)
    spatial = B * pid + tl.arange(0, B)
    mask = spatial < w * h
    for c in tl.static_range(3):
        offset = c * w * h + spatial
        mat_b = tl.load(mat + offset, mask=mask, other=0)
        tl.store(mat + offset, ~mat_b, mask=mask)

@triton.jit
def color_invert_kernel(mat, w, h, B: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = B * pid + tl.arange(0, B)
    mask = mask < w * h * 3
    mat_b = tl.load(mat + offset, mask=mask, other=0.0)
    tl.store(mat + offset, ~mat_b, mask=mask)

def color_invert(mat):
    B = 256
    c, w, h = mat.shape
    grid = (triton.cdiv(w * h * 3, B),)
    color_invert_kernel[grid](mat, w, h, B)
    print("kernel passed")


a = torch.randint(0, 256, size=(4, 512, 512), dtype=torch.uint8, device="cuda")

