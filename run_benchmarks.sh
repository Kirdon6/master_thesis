#!/bin/bash
#SBATCH --job-name=structure_prediction
#SBATCH --output=slurm_logs/%j.out       # Standard output log - %j is job ID
#SBATCH --error=slurm_logs/%j.err        # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Use 4 CPU cores
#SBATCH --mem=32G                       # Memory limit
#SBATCH --gres=gpu:4                    # Request 4 GPUs
#SBATCH --time=48:00:00                 # Time limit (48 hours)

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# IMPORTANT: Set CUDA_VISIBLE_DEVICES *before* activating Python environment
# Convert Slurm's GPU IDs into proper CUDA device indices (0,1,2,3)
if [ -n "$SLURM_JOB_GPUS" ]; then
    # This is a comma-separated list from Slurm like "0,1,2,3" or "1,2,3,4"
    echo "Slurm assigned GPUs: $SLURM_JOB_GPUS"
    
    # Always use sequential device IDs starting from 0, regardless of what Slurm assigns
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Now activate the environment AFTER setting CUDA_VISIBLE_DEVICES
source ~/master_thesis/nano_diff/bin/activate

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "# CPUs: $SLURM_CPUS_PER_TASK"

# Verify GPUs are available
nvidia-smi

# Specify a single config file to run
CONFIG_FILE="configs/diffusion_xpdf_abs_base.yaml"
echo "Using config file: $CONFIG_FILE"

# Extract model type from config file name
MODEL_TYPE="Unknown"
if [[ $CONFIG_FILE == *"diffusion"* ]]; then
    MODEL_TYPE="Diffusion"
elif [[ $CONFIG_FILE == *"mlp"* ]]; then
    MODEL_TYPE="MLP"
fi

echo "Running experiment with model type: $MODEL_TYPE"

# Run the experiment script with multi-GPU config
python run_benchmarks.py --config_path "$CONFIG_FILE" --use_multi_gpu False

deactivate

# Signal completion
echo "Job completed at $(date)" 