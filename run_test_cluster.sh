#!/bin/bash
#SBATCH --job-name=structure_prediction
#SBATCH --output=slurm_logs/%A_%a.out   # Standard output log - %A is job ID, %a is array task ID
#SBATCH --error=slurm_logs/%A_%a.err    # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Use 4 CPU cores
#SBATCH --mem=32G                       # Memory limit
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=48:00:00                 # Time limit (48 hours)
#SBATCH --array=0                     # Array job with 4 tasks (for different configs)

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Activate the virtual environment
source ~/master_thesis/nano_diff/bin/activate

# Print GPU information for debugging
echo "CUDA_VISIBLE_DEVICES before: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

# Important fix: Use 0 for CUDA_VISIBLE_DEVICES regardless of the GPU ID assigned by Slurm
# This tells PyTorch to use the first (and only) visible GPU
export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES after: $CUDA_VISIBLE_DEVICES"

# Print GPU information using nvidia-smi
nvidia-smi

# Run the experiment script
python test_torch_cluster.py

deactivate

# Signal completion
echo "Job completed at $(date)" 