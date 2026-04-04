from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def build_extension():
    this_dir = Path(__file__).resolve().parent
    return load(
        name="toy_mm80_ext",
        sources=[
            str(this_dir / "toy_mm_80.cu"),
            str(this_dir / "toy_mm_80_torch_binding.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3"],
        verbose=True,
    )

def compute_expected(a_in: torch.Tensor, b_in: torch.Tensor) -> torch.Tensor:
    assert a_in.device.type == "cuda" and b_in.device.type == "cuda"
    assert a_in.dtype == torch.float16 and b_in.dtype == torch.float16
    assert a_in.numel() >= 256 and b_in.numel() >= 128

    # map A -> (16,16)
    A = a_in[:256].reshape(16, 16).to(torch.float32)   # m x k

    # map B -> (16,8) (use first 128 elements)
    B = b_in[:128].reshape(16, 8).to(torch.float32)    # k x n

    # FP32 accumulation as in tensor cores
    C = torch.matmul(A, B)  # shape (16,8), dtype=float32

    # Flatten row-major and return the first 4 elements to match D[0..3]
    flat = C.reshape(-1)    # length 128
    return flat[:4].contiguous()


def calibrate_lane0_coords(ext) -> list[tuple[int, int]]:
    a_cal = torch.eye(16, device="cuda", dtype=torch.float16).reshape(-1).contiguous()
    b_mat = torch.zeros((16, 8), device="cuda", dtype=torch.float16)
    for k in range(16):
        for j in range(8):
            b_mat[k, j] = k * 16 + j
    b_cal = torch.zeros(256, device="cuda", dtype=torch.float16)
    b_cal[:128] = b_mat.reshape(-1)

    out_cal = ext.toy_mm80(a_cal, b_cal).to(torch.float32).cpu()
    c_cal = torch.matmul(
        a_cal[:256].reshape(16, 16).to(torch.float32),
        b_cal[:128].reshape(16, 8).to(torch.float32),
    ).cpu()

    value_to_coord = {float(c_cal[r, c].item()): (r, c) for r in range(16) for c in range(8)}
    coords: list[tuple[int, int]] = []
    for value in out_cal.tolist():
        key = float(value)
        if key not in value_to_coord:
            raise RuntimeError(f"Calibration failed: value {key} not found in reference matrix")
        coords.append(value_to_coord[key])
    return coords


def expected_from_coords(a_in: torch.Tensor, b_in: torch.Tensor, coords: list[tuple[int, int]]) -> torch.Tensor:
    c = torch.matmul(
        a_in[:256].reshape(16, 16).to(torch.float32),
        b_in[:128].reshape(16, 8).to(torch.float32),
    )
    return torch.tensor([c[r, col].item() for (r, col) in coords], device=a_in.device, dtype=torch.float32)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script")

    if torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("This kernel needs an SM80+ GPU")

    ext = build_extension()

    # Non-zero deterministic inputs.
    lane0_coords = calibrate_lane0_coords(ext)

    a = (torch.arange(256, device="cuda", dtype=torch.float16) % 19) * 0.25
    b = (torch.arange(256, device="cuda", dtype=torch.float16) % 13) * 0.5

    a_in = a.contiguous()
    b_in = b.contiguous()

    out = ext.toy_mm80(a_in, b_in)
    expected = expected_from_coords(a_in, b_in, lane0_coords)

    torch.cuda.synchronize()

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    a_row0 = a_in[:16].to(torch.float32).cpu()
    b_rows = b_in[:128].reshape(16, 8).to(torch.float32).cpu()
    out_cpu = out.cpu()
    expected_cpu = expected.cpu()

    print("Lane0 fragment maps to C coords:", lane0_coords)
    print("A row0 (16 values):", a_row0.tolist())
    print("B col0..3 by k (16x4):", b_rows[:, :4].tolist())
    print("Expected D[0..3]:", expected_cpu.tolist())
    print("Kernel   D[0..3]:", out_cpu.tolist())
    print("Abs diff         :", (out_cpu - expected_cpu).abs().tolist())


if __name__ == "__main__":
    main()
