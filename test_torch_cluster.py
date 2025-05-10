import torch
import os

print("\n===== CUDA Environment =====")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS', 'Not set')}")

print("\n===== CUDA Availability =====")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Get properties for each device
    for i in range(torch.cuda.device_count()):
        print(f"\n===== Device {i} =====")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
    # Try to create a tensor on GPU
    try:
        print("\n===== GPU Tensor Test =====")
        x = torch.ones(10).cuda()
        print(f"Successfully created tensor on GPU: {x.device}")
    except Exception as e:
        print(f"Failed to create tensor on GPU: {e}")
else:
    print("\nCUDA is not available. Check your PyTorch installation or GPU drivers.")