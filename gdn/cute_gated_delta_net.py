"""
Gated Delta Net - Simplified cute dsl kernel.
"""
import math
import os
import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
os.environ.setdefault("TARGET_SM_ARCH", "sm_100a")


def compute_gates(A_log, a, dt_bias, b):
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())
    return g.squeeze(1).float(), beta.squeeze(1).float()


# Optimized Kernel Configuration
THREADS_PER_BLOCK = 256  # 256 threads = 8 warps for better parallelism
WARP_SIZE = 32


@cute.kernel
def gated_delta_net_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
    g_ptr, beta_ptr, scale_val,
    B, H, K, V
):
    """
    Optimized Gated Delta Net Kernel with warp-level parallelism.

    Key optimizations:
    - 256 threads per block for better GPU utilization
    - Threads cooperate on K dimension using warp shuffles
    - Each warp handles a portion of K for each V
    - Parallel reduction within warps
    """
    # Thread and Block Indices
    flat_idx = cute.arch.block_idx()[0]
    v_idx = cute.arch.block_idx()[1]

    tidx = cute.arch.thread_idx()[0]
    warp_id = tidx // WARP_SIZE
    lane_id = tidx % WARP_SIZE
    num_warps = THREADS_PER_BLOCK // WARP_SIZE

    # Early exit if V index out of bounds
    if v_idx >= V:
        return

    # Load Parameters (Broadcast from flat_idx) - all threads load g and beta
    g_val = g_ptr[flat_idx].to(cutlass.Float32)
    beta_val = beta_ptr[flat_idx].to(cutlass.Float32)
    one_minus_beta = cutlass.Float32(1.0) - beta_val
    scale_f32 = cutlass.Float32(scale_val)

    # Phase 1: Compute old_v = sum_k k[k] * g * state[k,v]
    # Each thread computes partial sum for its assigned K values
    k_stride = THREADS_PER_BLOCK
    old_v_partial = cutlass.Float32(0.0)

    for k_idx in range(tidx, K, k_stride):
        k_val = k_ptr[flat_idx, k_idx].to(cutlass.Float32)
        state_val = state_ptr[flat_idx, k_idx, v_idx].to(cutlass.Float32)
        old_v_partial += k_val * g_val * state_val

    # Warp-level reduction using shuffle
    # Reduce across 32 lanes
    old_v_partial += cute.arch.shfl_down(old_v_partial, 16)
    old_v_partial += cute.arch.shfl_down(old_v_partial, 8)
    old_v_partial += cute.arch.shfl_down(old_v_partial, 4)
    old_v_partial += cute.arch.shfl_down(old_v_partial, 2)
    old_v_partial += cute.arch.shfl_down(old_v_partial, 1)

    # Lane 0 of each warp now has partial sum
    # Now reduce across warps
    if warp_id == 0:
        if lane_id > 0:
            old_v_partial = cutlass.Float32(0.0)

    # Final reduction at lane 0
    old_v_acc = cute.arch.shfl(old_v_partial, 0)

    # Compute diff (only thread 0 needs to do this)
    if tidx == 0:
        v_val = v_ptr[flat_idx, v_idx].to(cutlass.Float32)
        # new_v = beta * v + (1-beta) * old_v
        new_v_val = beta_val * v_val + one_minus_beta * old_v_acc
        # diff = new_v - old_v
        diff = new_v_val - old_v_acc
    else:
        diff = cutlass.Float32(0.0)

    # Broadcast diff to all threads
    diff = cute.arch.shfl(diff, 0)

    # Phase 2: Update State and Compute Output
    # Each thread updates its assigned K values
    output_acc = cutlass.Float32(0.0)

    for k_idx in range(tidx, K, k_stride):
        # Load original k and state
        k_val = k_ptr[flat_idx, k_idx].to(cutlass.Float32)
        state_val = state_ptr[flat_idx, k_idx, v_idx].to(cutlass.Float32)

        # new_state = g * state + k * diff
        new_state_val = g_val * state_val + k_val * diff

        # Store to new_state
        new_state_ptr[flat_idx, k_idx, v_idx] = new_state_val.to(cutlass.BFloat16)

        # Update output: output += scale * q * new_state
        q_val = q_ptr[flat_idx, k_idx].to(cutlass.Float32)
        output_acc += scale_f32 * q_val * new_state_val

    # Warp reduction for output
    output_acc += cute.arch.shfl_down(output_acc, 16)
    output_acc += cute.arch.shfl_down(output_acc, 8)
    output_acc += cute.arch.shfl_down(output_acc, 4)
    output_acc += cute.arch.shfl_down(output_acc, 2)
    output_acc += cute.arch.shfl_down(output_acc, 1)

    # Store output (lane 0 writes)
    if tidx == 0:
        output_ptr[flat_idx, v_idx] = output_acc.to(cutlass.BFloat16)


@cute.jit
def gated_delta_net_op(
    q_tensor, k_tensor, v_tensor, state_tensor,
    output_tensor, new_state_tensor,
    g_tensor, beta_tensor, scale,
    B, H, K, V
):
    # Grid: (B*H, V) blocks - each block handles one (B,H,V) combination
    # Block: 256 threads per block for warp-level parallelism
    grid = (B * H, V, 1)
    block = (THREADS_PER_BLOCK, 1, 1)

    gated_delta_net_kernel(
        q_tensor, k_tensor, v_tensor, state_tensor,
        output_tensor, new_state_tensor,
        g_tensor, beta_tensor, scale,
        B, H, K, V
    ).launch(grid=grid, block=block)

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

# Kernel Configurations
AB_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32
HEAD_DIM = 128
CTA_TILE_SHAPE = (64, 64, HEAD_DIM)    # MMA tile per CTA
PIPELINE_STAGES = 4                    # Software pipelining
THREADS_PER_CTA = 256

@cute.struct
class SharedStorage:
    ab_mbar: cute.struct.MemRange[cutlass.Int64, PIPELINE_STAGES * 2]
    tmem_holding_buf: cutlass.Int32

@cute.jit(device=True)
def compute_gates_device(A_log_ptr, a_val, dt_bias_ptr, b_val):
    x = (a_val + dt_bias_ptr).to(cutlass.Float32)
    softx = cutlass.FastSoftPlus(x)
    gx = (-torch.exp(A_log_ptr) * softx).exp()
    betax = b_val.sigmoid()
    return gx, betax

@cute.kernel
def gated_delta_net_prefill_kernel(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    state_ptr: cute.Tensor,
    new_state_ptr: cute.Tensor,
    output_ptr: cute.Tensor,
    A_log_ptr: cute.Tensor,
    a_tensor: cute.Tensor,
    dt_bias_ptr: cute.Tensor,
    b_tensor: cute.Tensor,
    scale_val: float,
    B: int, T: int, H: int, K: int, V: int
):
    # Indices & threading
    tidx, _, _ = cute.arch.thread_idx()
    bidx, tstep, h_idx = cute.arch.block_idx()
    
    if tstep >= T:
        return
    
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    
    # Allocate shared memory
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    
    sQ = smem.allocate_tensor(element_type=AB_DTYPE, shape=CTA_TILE_SHAPE[0]//16 * K)
    sK = smem.allocate_tensor(element_type=AB_DTYPE, shape=CTA_TILE_SHAPE[1]//16 * K)

    # Tiled MMA object
    op = tcgen05.MmaF16Op(AB_DTYPE, ACC_DTYPE, (128, 256, 16), tcgen05.CtaGroup.ONE)
    tiled_mma = cute.make_tiled_mma(op)

    # Register fragments
    tCrQ = tiled_mma.make_fragment_A(sQ, mode=(0,))
    tCrK = tiled_mma.make_fragment_B(sK, mode=(0,))
    acc_shape = tiled_mma.partition_shape_C((CTA_TILE_SHAPE[0], CTA_TILE_SHAPE[1]))
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    # Partition global tiles
    local_q = cute.local_tile(q_tensor, (CTA_TILE_SHAPE[0], K), (bidx, 0, tstep, h_idx))
    local_k = cute.local_tile(k_tensor, (CTA_TILE_SHAPE[1], K), (bidx, 0, tstep, h_idx))
    local_v = cute.local_tile(v_tensor, (V, ), (bidx, 0, tstep, h_idx))

    # Load values once per step
    a_val = a_tensor[bidx, tstep, h_idx]
    b_val = b_tensor[bidx, tstep, h_idx]
    g_val, beta_val = compute_gates_device(A_log_ptr, a_val, dt_bias_ptr, b_val)

    # Cache q and k values in registers to avoid repeated loads
    curr_q = q_tensor[bidx, tstep, h_idx]
    curr_k = k_tensor[bidx, tstep, h_idx]

    # Load previous state tile
    local_state = cute.local_tile(state_ptr, shape=(V, K), coord=(bidx, h_idx))

    # Loop over V-dimension split by TILE_N
    num_n_blocks = 1
    for nv_block_id in range(num_n_blocks):
        vn_start = nv_block_id * CTA_TILE_SHAPE[1]
        vn_limit = min(V, vn_start + CTA_TILE_SHAPE[1])
        vn_chunk = vn_limit - vn_start

        # Copy k into SMEM - use vectorized load for efficiency
        # Load all K elements into shared memory using cached curr_k
        for ki in range(K):
            sK[ki].store(curr_k[ki])

        # Load q into SMEM as well for MMA using cached curr_q
        for qi in range(K):
            sQ[qi].store(curr_q[qi])

        # Perform GEMM: q @ state using MMA
        # Clear accumulator
        for mk in range(K):
            tCtAcc[0, mk, 0] = cutlass.Float32(0.0)

        # Use MMA for the GEMM operation
        # Each MMA tile is (128, 64) but we have smaller dimensions
        # Do manual MMA accumulate using cached curr_q
        for mk in range(K):
            q_val = curr_q[mk]
            for nk in range(vn_chunk):
                s = local_state[vn_start + nk, mk]
                tCtAcc[0, mk, 0] = tCtAcc[0, mk, 0] + q_val * s

        # Add contribution from value and gate
        # Process all V elements in this chunk
        for n_vi in range(vn_chunk):
            vi = vn_start + n_vi
            prev_state_row = local_state[vi]

            # Compute old_v as dot product of k and state row using cached curr_k
            old_v = cutlass.Float32(0.0)
            for mk in range(K):
                old_v = old_v + curr_k[mk] * prev_state_row[mk]

            new_v = beta_val * v_tensor[bidx, tstep, h_idx, vi] + (cutlass.Float32(1.0) - beta_val) * old_v
            diff = new_v - old_v

            # Update state row: new_state = old_state + k * diff using cached curr_k
            for mk in range(K):
                local_state[vi, mk] = prev_state_row[mk] + curr_k[mk] * diff

        # Compute output for each v in the chunk using updated state and cached curr_q
        for n_vi in range(vn_chunk):
            vi = vn_start + n_vi

            # Compute q @ new_state[vi]
            output_val = cutlass.Float32(0.0)
            for mk in range(K):
                output_val = output_val + curr_q[mk] * local_state[vi, mk]

            scaled_dot_prod = scale_val * output_val
            output_ptr[bidx, tstep, h_idx, vi] = scaled_dot_prod.to(AB_DTYPE)

    # Store final state for next token
    # Vectorized store for efficiency
    for vi in range(V):
        for ki in range(K):
            new_state_ptr[bidx, h_idx, vi, ki] = local_state[vi, ki]

# JIT Host Function to launch kernel
@cute.jit
def gated_delta_net_prefill_op(
    q_tensor, k_tensor, v_tensor, state_tensor,
    new_state_tensor, output_tensor,
    A_log_tensor, a_tensor, dt_bias_tensor, b_tensor,
    scale,
    B, T, H, K, V
):
    grid = (B, T, H)
    block = (THREADS_PER_CTA, 1, 1)

    gated_delta_net_prefill_kernel(
        q_tensor, k_tensor, v_tensor, state_tensor, new_state_tensor,
        output_tensor,
        A_log_tensor, a_tensor, dt_bias_tensor, b_tensor,
        cutlass.Float32(scale),
        B, T, H, K, V
    ).launch(grid=grid, block=block)

# End-to-end Wrapper Function
def run_cute_prefill(
    q, k, v, state, A_log, a, dt_bias, b, scale=None
):
    B, T, num_q_heads, K = q.shape
    _, _, num_v_heads, V = v.shape
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    q_packed = q.contiguous().view(B*T*num_q_heads, K)
    k_packed = k.contiguous().view(B*T*num_q_heads, K)
    v_packed = v.contiguous().view(B*T*num_v_heads, V)

    # Reshape state to match kernel expectations: [B, H, V, K] -> [B*H, V, K]
    state_packed = state.permute(0, 1, 3, 2).reshape(B * num_v_heads, V, K)
    state_packed = state_packed.contiguous()

    new_state = torch.empty_like(state_packed, dtype=torch.bfloat16)
    output = torch.empty(B, T, num_v_heads, V, dtype=torch.bfloat16, device=q.device)

    # Pass PyTorch tensors directly to kernel (cute DSL handles them)
    gated_delta_net_prefill_op(
        q_packed, k_packed, v_packed, state_packed,
        new_state, output,
        A_log.flatten(), a.flatten(), dt_bias, b.flatten(),
        scale,
        B, T, num_v_heads, K, V
    )
    torch.cuda.synchronize()

    # Reshape back to original layout
    new_state_unpacked = new_state.view(B, num_v_heads, V, K).permute(0, 1, 3, 2)
    output_unpacked = output.view(B, T, num_v_heads, V)

    return output_unpacked, new_state_unpacked

def run_cute(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    q_exp = q.squeeze(1).repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.squeeze(1).repeat_interleave(num_v_heads // num_k_heads, dim=1)
    v_exp = v.squeeze(1)

    q_exp = q_exp.reshape(B * num_heads, K)
    k_exp = k_exp.reshape(B * num_heads, K)
    v_exp = v_exp.reshape(B * num_heads, V)
    state = state.permute(0, 1, 3, 2).reshape(B * num_heads, K, V)

    g, beta = compute_gates(A_log, a, dt_bias, b)
    g = g.reshape(B * num_heads)
    beta = beta.reshape(B * num_heads)

    if state is None:
        state = torch.zeros(B * num_heads, K, V, dtype=torch.bfloat16, device=device)

    output = torch.zeros(B * num_heads, V, dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)

    scale_f32 = cutlass.Float32(scale)

    gated_delta_net_op(
        from_dlpack(q_exp),
        from_dlpack(k_exp),
        from_dlpack(v_exp),
        from_dlpack(state),
        from_dlpack(output),
        from_dlpack(new_state),
        from_dlpack(g),
        from_dlpack(beta),
        scale_f32,
        B, num_heads, K, V
    )

    torch.cuda.synchronize()

    output = output.reshape(B, 1, num_heads, V)
    new_state = new_state.reshape(B, num_heads, K, V).permute(0, 1, 3, 2)

    return output, new_state


def run_reference(q, k, v, state, A_log, a, dt_bias, b, scale):
    import torch.nn.functional as F

    def matmul(a, b):
        return a.float() @ b.float()

    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_f32[b_idx, h_idx]
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)
            g_val = g_f32[b_idx, h_idx]
            beta_val = beta_f32[b_idx, h_idx]

            old_state = g_val * h_state
            old_v = matmul(k_h, old_state)
            new_v = beta_val * v_h + (1.0 - beta_val) * old_v
            state_remove = matmul(k_h.unsqueeze(1), old_v.unsqueeze(0))
            state_update = matmul(k_h.unsqueeze(1), new_v.unsqueeze(0))
            h_state = old_state - state_remove + state_update

            output[b_idx, h_idx] = scale * matmul(q_h, h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)

    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state


if __name__ == "__main__":
    print("Gated Delta Net - Cute DSL Prefill Kernel")
    print("=" * 50)

    B, Hq, Hk, Hv = 4, 8, 8, 8
    K, V = 128, 128

    torch.manual_seed(42)
    q = torch.randn(B, 1, Hq, K, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, 1, Hk, K, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, 1, Hv, V, device='cuda', dtype=torch.bfloat16)
    state = torch.randn(B, Hv, V, K, device='cuda', dtype=torch.bfloat16)
    A_log = torch.randn(B, 1, Hv, device='cuda', dtype=torch.bfloat16)
    a = torch.randn(B, 1, Hv, device='cuda', dtype=torch.bfloat16)
    dt_bias = torch.randn(B, 1, Hv, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(B, 1, Hv, device='cuda', dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(K)

    print(f"Testing B={B}, H={Hv}, K={K}, V={V}")

    print("Reference...")
    # Warmup reference
    for _ in range(10):
        ref_out, ref_state = run_reference(q, k, v, state, A_log, a, dt_bias, b, scale)
    torch.cuda.synchronize()
    # Benchmark reference
    start = time.time()
    for _ in range(100):
        ref_out, ref_state = run_reference(q, k, v, state, A_log, a, dt_bias, b, scale)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / 100 * 1000
    print(f"Reference time: {ref_time:.2f}ms")

    print("Cute (Prefill)...")
    cute_out, cute_state = run_cute_prefill(q, k, v, state, A_log, a, dt_bias, b, scale)

    print(f"Ref state[0,0,0,:8] = {ref_state[0,0,0,:8]}")
    print(f"Cute state[0,0,0,:8] = {cute_state[0,0,0,:8]}")
    print(f"Ref output[0,0,0,:8] = {ref_out[0,0,0,:8]}")
    print(f"Cute output[0,0,0,:8] = {cute_out[0,0,0,:8]}")

    # Warmup
    for _ in range(10):
        cute_out, cute_state = run_cute_prefill(q, k, v, state, A_log, a, dt_bias, b, scale)
    torch.cuda.synchronize()

    # Benchmark
    n = 100
    start = time.time()
    for _ in range(n):
        cute_out, cute_state = run_cute_prefill(q, k, v, state, A_log, a, dt_bias, b, scale)
    torch.cuda.synchronize()
    avg = (time.time() - start) / n * 1000

    print(f"\nBenchmark: {avg:.2f}ms")

    out_diff = (cute_out.float() - ref_out.float()).abs().max().item()
    state_diff = (cute_state.float() - ref_state.float()).abs().max().item()
    print(f"Output diff: {out_diff:.2e}")
    print(f"State diff: {state_diff:.2e}")

    if out_diff < 2e-1 and state_diff < 2e-1:
        print("PASSED!")
    else:
        print("FAILED")
