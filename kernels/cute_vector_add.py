import cutlass.cute as cute
import torch
import cutlass
from cutlass import Float32
import cutlass.cute.testing
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def matrix_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, m: cutlass.Int32, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    bidx, _, _ = cute.arch.block_idx()

    idx = bidx * bdim + tidx

    global_row = block_row * block_dim_row + row
    global_col = block_col * block_dim_col + col

    if global_row < m and global_col < n:
        a_val = a[global_row, global_col]
        b_val = b[global_row, global_col]
        c[global_row, global_col] = a_val + b_val

@cute.jit
def matrix_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    m, n = a.shape

    grid_dim = ( (m + 15) // 16, (n + 15) // 16, 1 )
    block_dim = (16, 16, 1)

    matrix_add_kernel(a, b, c, cutlass.Int32(m), cutlass.Int32(n)).launch(grid=grid_dim, block=block_dim)


if __name__ == "__main__":
    m, n = 1024, 1024
    a = torch.randn((m, n), dtype=torch.float16, device='cuda')
    b = torch.randn((m, n), dtype=torch.float16, device='cuda')
    c = torch.empty((m, n), dtype=torch.float16, device='cuda')
    a_cute = from_dlpack(a, assumed_align=32)
    b_cute = from_dlpack(b, assumed_align=32)
    c_cute = from_dlpack(c, assumed_align=32)

    def kernel_valid(kernel):
        c.zero_()
        kernel(a_cute, b_cute, c_cute)
        torch.testing.assert_close(c, a + b)

    def bench_kernel(kernel):
        c.zero_()
        avg_ms = cutlass.cute.testing.benchmark(
            kernel,
            kernel_arguments=cutlass.cute.testing.JitArguments(a_cute, b_cute, c_cute),
            warmup_iterations=10,
            iterations=100,
        )
        print(f"Average execution time: {avg_ms:.3f} ms")


    kernel_valid(matrix_add)
    bench_kernel(matrix_add)