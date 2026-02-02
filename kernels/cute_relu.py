import cutlass
from cutlass import Float32
import cutlass.cute as cute
import cutlass.cute.testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
import time
import torch

@dsl_user_op
def relu_op(x: cute.Tensor, **kwargs):
    return cute.arch.fmax(x, Float32(0.0))

@cute.kernel
def relu_kernel_naive(input_tensor: cute.Tensor, output_tensor: cute.Tensor, size: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < size:
        x = input_tensor[idx]
        x_float = x.to(cute.Float32)
        result = relu_op(x_float)
        output_tensor[idx] = result.to(output_tensor.element_type)


@cute.kernel
def relu_kernel_vectorized(gInput: cute.Tensor, gOutput: cute.Tensor, num_vectors: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    
    if thread_idx < num_vectors:
        input_vec = gInput[(None, thread_idx)].load()
        
        v0 = relu_op(input_vec[0])
        v1 = relu_op(input_vec[1])
        v2 = relu_op(input_vec[2])
        v3 = relu_op(input_vec[3])
        
        gOutput[(0, thread_idx)] = v0.to(gOutput.element_type)
        gOutput[(1, thread_idx)] = v1.to(gOutput.element_type)
        gOutput[(2, thread_idx)] = v2.to(gOutput.element_type)
        gOutput[(3, thread_idx)] = v3.to(gOutput.element_type)

@cute.kernel
def relu_kernel_kv(input_tensor: cute.Tensor, output_tensor: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # For 1D tiled tensor: (tile_shape, num_tiles)
    blk_coord = (None, bidx)

    blkA = input_tensor[blk_coord]
    blkB = output_tensor[blk_coord]

    tvA = cute.composition(blkA, tv_layout)
    tvB = cute.composition(blkB, tv_layout)

    thr_coord = (tidx, None)
    tvA_thread = tvA[thr_coord]
    tvB_thread = tvB[thr_coord]

    # Apply ReLU element-wise - iterate through each element
    # tvA_thread has shape (8,) based on TV layout (256, 8)
    for i in range(8):
        tvB_thread[i] = relu_op(tvA_thread[i]).to(output_tensor.element_type)


@cute.jit
def relu_naive(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    num_threads_per_block = 256
    size = input_tensor.shape[0]

    kernel = relu_kernel_naive(input_tensor, output_tensor, cutlass.Int32(size))

    kernel.launch(
        grid=((size + num_threads_per_block - 1) // num_threads_per_block, 1, 1),
        block=(num_threads_per_block, 1, 1)
    )


@cute.jit
def relu_vectorized(mInput: cute.Tensor, mOutput: cute.Tensor):
    threads_per_block = 64

    # Partition input tensor into groups of 4 contiguous elements
    # For 1D tensor of shape (N,), this creates ((4,), (N/4,)) : ((1,), (4,))
    gInput = cute.zipped_divide(mInput, (4,))
    gOutput = cute.zipped_divide(mOutput, (4,))

    # Number of vectors (each vector has 4 elements)
    num_vectors = cute.size(gInput, mode=[1])
    grid_size = (num_vectors + threads_per_block - 1) // threads_per_block

    relu_kernel_vectorized(gInput, gOutput, cutlass.Int32(num_vectors)).launch(
        grid=(grid_size, 1, 1),
        block=(threads_per_block, 1, 1)
    )


@cute.jit
def relu_kv(mInput: cute.Tensor, mOutput: cute.Tensor):
    # For 1D tensor, use TV layout with thread and value dimensions
    # Each thread loads coalesced_ldst_bytes (16 bytes)
    coalesced_ldst_bytes = 16
    
    # Compile time validation: expect same element type for input and output
    assert mInput.element_type == mOutput.element_type
    dtype = mInput.element_type
    
    # Thread layout: 256 threads per block
    # Value layout: each thread processes coalesced_ldst_bytes worth of elements
    thr_layout = cute.make_ordered_layout((256,), order=(0,))
    val_layout = cute.make_ordered_layout((coalesced_ldst_bytes,), order=(0,))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    # print(f"[DSL INFO] Tiler: {tiler}")
    # print(f"[DSL INFO] TV Layout: {tv_layout}")
    
    # Tile the 1D tensors
    gInput = cute.zipped_divide(mInput, tiler)
    gOutput = cute.zipped_divide(mOutput, tiler)
    
    # print("[DSL INFO] Tiled Tensors:")
    # print(f"[DSL INFO]   gInput = {gInput.type}")
    # print(f"[DSL INFO]   gOutput = {gOutput.type}")
    
    # Launch kernel
    relu_kernel_kv(gInput, gOutput, tv_layout).launch(
        grid=[cute.size(gInput, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


if __name__ == "__main__":
    inp = torch.randn(1 << 20, device='cuda').to(torch.bfloat16)
    out = torch.empty_like(inp)
    input_cute = from_dlpack(inp, assumed_align=16)
    output_cute = from_dlpack(out, assumed_align=16)

    def verify_relu(fn_name: str, fn):
        out.zero_()
        fn(input_cute, output_cute)
        torch.testing.assert_close(out, inp.clamp(min=0))
        print(f"{fn_name} output matches torch.relu")


    def benchmark_relu(fn_name: str, fn):
        out.zero_()
        avg_time_us = cute.testing.benchmark(
            fn,
            kernel_arguments=cute.testing.JitArguments(input_cute, output_cute),
            warmup_iterations=5,
            iterations=100,
        )
        print(f"{fn_name} avg time: {avg_time_us:.2f} us")
        return avg_time_us


    verify_relu("relu_naive", relu_naive)
    verify_relu("relu_vectorized", relu_vectorized)
    verify_relu("relu_kv", relu_kv)

    benchmark_relu("relu_naive", relu_naive)
    benchmark_relu("relu_vectorized", relu_vectorized)
    benchmark_relu("relu_kv", relu_kv)
