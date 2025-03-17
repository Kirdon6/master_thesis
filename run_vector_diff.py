import os
import torch
from benchmark import run_benchmark_from_notebook

# Check if CUDA is available
use_cpu = not torch.cuda.is_available()
if use_cpu:
    print("CUDA not available, using CPU")
else:
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

# Run the benchmark
config_path = "vector_diff_config.yaml"
print(f"Running benchmark with config: {config_path}")
results = run_benchmark_from_notebook(config_path, use_cpu=use_cpu)

# Print results
print("\nBenchmark Results:")
print(results) 