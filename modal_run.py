#!/usr/bin/env python3
import sys
import modal
from modal.mount import Mount
import os
import subprocess
from enum import Enum
from pathlib import Path

class GPUType(Enum):
    T4 = "T4"
    L4 = "L4"
    A10 = "A10"
    A100 = "A100"
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    L40S = "L40S"
    H100 = "H100"
    H200 = "H200"
    B200 = "B200"

# Parameters for run
DEFAULT_GPU = "H100"
DEFAULT_TIMEOUT_MINS = 2
DEFAULT_SCRIPT = "rms_norm.py"

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "nvidia-cutlass-dsl==4.3.5",
        "triton",
        "ninja",
    )
    .add_local_dir("kernels/", remote_path="/root/scripts")
)

app = modal.App("cute-reduction")

@app.local_entrypoint()
def main(script: str = DEFAULT_SCRIPT, gpu: str = DEFAULT_GPU, timeout: int = DEFAULT_TIMEOUT_MINS):
    """
    Run any Python script on Modal with GPU support.

    Args:
        script: Path to Python script to execute (default: rms_norm.py)
        gpu: GPU type (e.g., 'H100', 'A100', 'T4')
        timeout: Timeout in minutes

    Example:
        modal run run_rms_norm.py --script rms_norm.py --gpu H100 --timeout 10
        modal run run_rms_norm.py --script reduction_cute.py --gpu A100 --timeout 5
        modal run run_rms_norm.py  # defaults: rms_norm.py, H100, 10min
    """
    if not os.path.exists(script):
        print(f"Error: Script '{script}' not found")
        sys.exit(1)

    script_name = os.path.basename(script)
    print(f"Configuring Modal: script={script_name}, GPU={gpu}, timeout={timeout}min")
    execute.remote(script_name, gpu, timeout)

@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_MINS * 60
)
def execute(script: str, gpu: str, timeout: int):
    """Execute any Python script with specified GPU and timeout."""
    import os
    from pathlib import Path

    script_name = os.path.basename(script)
    file_ext = Path(script_name).suffix.lower()

    print(f"Executing {script_name} with GPU={gpu}, timeout={timeout}min")

    os.environ['GPU'] = gpu
    os.environ['TIMEOUT'] = str(timeout * 60)

    os.chdir("/root/scripts")

    if file_ext == ".cu":
        result = compile_and_run_cuda(script_name, gpu)
    else:  # .py files
        result = os.system(f"python {script_name}")

    if result != 0:
        print(f"Script exited with code {result}")
    else:
        print(f"Script completed successfully")


def compile_and_run_cuda(cuda_file: str, gpu: str, nvcc_args: list[str] = None):
    import os
    import subprocess

    if nvcc_args is None:
        nvcc_args = []

    base_name = os.path.splitext(cuda_file)[0]
    output_binary = f"{base_name}"

    print(f"Compiling CUDA file: {cuda_file}")
    # print(f"Output binary: {output_binary}")

    nvcc_cmd = ["nvcc", cuda_file, "-o", output_binary] + nvcc_args

    try:
        result = subprocess.run(nvcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return result.returncode

        # print(f"Compilation successful")

        print(f"Running {output_binary}")
        result = subprocess.run([f"./{output_binary}"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors:\n{result.stderr}")

        return result.returncode

    except Exception as e:
        print(f"Error during CUDA compilation/execution: {e}")
        return 1

