#!/bin/bash
#SBATCH --job-name=structure_prediction
#SBATCH --output=slurm_logs/%A_%a.out   # Standard output log - %A is job ID, %a is array task ID
#SBATCH --error=slurm_logs/%A_%a.err    # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Use 4 CPU cores
#SBATCH --mem=32G                       # Memory limit
#SBATCH --gres=gpu:4                    # Request 4 GPUs
#SBATCH --time=48:00:00                 # Time limit (48 hours)
#SBATCH --array=0-3                     # Array job with 4 tasks (for different configs)

source ~/master_thesis/nano_diff/bin/activate

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "# CPUs: $SLURM_CPUS_PER_TASK"

# Print GPU environment variables before fixing
echo "CUDA_VISIBLE_DEVICES before: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

# Important fix: Set CUDA_VISIBLE_DEVICES to use indices 0-3 regardless of Slurm's assigned IDs
# This tells PyTorch to use the GPUs starting from index 0
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "CUDA_VISIBLE_DEVICES after: $CUDA_VISIBLE_DEVICES"
echo "Available GPU(s):"
nvidia-smi

# Define configuration files for different tasks
CONFIG_FILES=(
    "configs/diffusion_xpdf_abs_base.yaml"  # Task 0: Diffusion model with xPDF
    "configs/diffusion_xrd_abs_base.yaml"   # Task 1: Diffusion model with XRD
    "configs/mlp_xpdf_abs_base.yaml"        # Task 2: MLP model with xPDF
    "configs/mlp_xrd_abs_base.yaml"         # Task 3: MLP model with XRD
)

# Select the config file based on array task ID
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Using config file: $CONFIG_FILE"

# Extract model type from config file name
MODEL_TYPE="Unknown"
if [[ $CONFIG_FILE == *"diffusion"* ]]; then
    MODEL_TYPE="Diffusion"
elif [[ $CONFIG_FILE == *"mlp"* ]]; then
    MODEL_TYPE="MLP"
fi

echo "Running experiment with model type: $MODEL_TYPE"

# Run the experiment script
python run_benchmarks.py --config_path "$CONFIG_FILE"

deactivate

# Signal completion
echo "Job completed at $(date)" 