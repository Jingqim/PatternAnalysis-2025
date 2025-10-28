# ...existing code...
import sys
try:
    import torch
except Exception as e:
    print("PyTorch not installed:", e, file=sys.stderr)
    sys.exit(1)

print("PyTorch version:", torch.__version__)
print("CUDA toolkit version detected by torch:", torch.version.cuda)
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    count = torch.cuda.device_count()
    print("CUDA device count:", count)
    for i in range(count):
        try:
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {name}")
            print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print(f"  Compute capability: {props.major}.{props.minor}")
        except Exception as e:
            print(f"  Failed to query device {i}: {e}")
else:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print("nvidia-smi output:\n", out.stdout)
    except Exception:
        print("nvidia-smi not available or no NVIDIA driver detected.")
# ...existing code...




import torch
print("torch.version =", torch.version)      # e.g. 2.4.0+cu121 or 2.4.0+cpu
print("torch.version.cuda =", torch.version.cuda)    # e.g. '12.1' (None for CPU-only)
print("CUDA available?     =", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1, device="cuda")
    print("Tensor device:", x.device)