#!/usr/bin/env python3
import sys
import modal
from modal.mount import Mount
import os

# Parameters for run
DEFAULT_GPU = "H100"
DEFAULT_TIMEOUT_MINS = 2
DEFAULT_SCRIPT = "rms_norm.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "gcc")
    .pip_install(
        "torch",
        "numpy",
        "nvidia-cutlass-dsl==4.3.5",
        "triton"
    )
    .env({"CUDA_TOOLKIT_PATH": "/usr/local/cuda"})
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
    
    print(f"Configuring Modal: script={script}, GPU={gpu}, timeout={timeout}min")
    execute.remote(script, gpu, timeout)

@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_MINS * 60
)
def execute(script: str, gpu: str, timeout: int):
    """Execute any Python script with specified GPU and timeout."""
    import os
    
    print(f"Using remote python version:")
    os.system("python --version")
    
    script_name = os.path.basename(script)
    print(f"Executing {script_name} with GPU={gpu}, timeout={timeout}min")
    
    # Set environment variables for the script to use
    os.environ['GPU'] = gpu
    os.environ['TIMEOUT'] = str(timeout * 60)
    
    # Execute the script
    os.chdir("/root/scripts")
    result = os.system(f"python {script_name}")
    
    if result != 0:
        print(f"Script exited with code {result}")
    else:
        print(f"Script completed successfully")

