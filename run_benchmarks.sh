#!/bin/bash
#SBATCH --job-name=structure_prediction
#SBATCH --output=slurm_logs/%j_%x.out   # Standard output log - %j is job ID, %x is job name
#SBATCH --error=slurm_logs/%j_%x.err    # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Use 4 CPU cores
#SBATCH --mem=32G                       # Memory limit
#SBATCH --gres=gpu:4                    # Request all 4 GPUs
#SBATCH --time=48:00:00                 # Time limit (48 hours)

source ~/master_thesis/nano_diff/bin/activate

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "# CPUs: $SLURM_CPUS_PER_TASK"

# Print GPU environment variables
echo "CUDA_VISIBLE_DEVICES before: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

# Make all GPUs visible, using indices 0-3
export CUDA_VISIBLE_DEVICES=0,1,2,3


echo "CUDA_VISIBLE_DEVICES after: $CUDA_VISIBLE_DEVICES"
echo "Available GPU(s):"
nvidia-smi

# Define configuration files
CONFIG_FILES=(
    "configs/diffusion_xpdf_abs_base.yaml"  # Diffusion model with xPDF
    "configs/diffusion_xrd_abs_base.yaml"   # Diffusion model with XRD
    "configs/mlp_xpdf_abs_base.yaml"        # MLP model with xPDF
    "configs/mlp_xrd_abs_base.yaml"         # MLP model with XRD
)

# Run each configuration sequentially
for ((i=0; i<${#CONFIG_FILES[@]}; i++)); do
    CONFIG_FILE=${CONFIG_FILES[$i]}
    echo "=============================================="
    echo "Running job $((i+1)) of ${#CONFIG_FILES[@]}: $CONFIG_FILE"
    echo "=============================================="

    # Extract model type from config file name
    MODEL_TYPE="Unknown"
    if [[ $CONFIG_FILE == *"diffusion"* ]]; then
        MODEL_TYPE="Diffusion"
    elif [[ $CONFIG_FILE == *"mlp"* ]]; then
        MODEL_TYPE="MLP"
    fi

    echo "Running experiment with model type: $MODEL_TYPE"
    
    # Run the experiment script
    python run_benchmarks.py --config_path "$CONFIG_FILE" --task_id $i
    
    echo "Completed job $((i+1)): $CONFIG_FILE at $(date)"
    echo ""
done

deactivate

# Signal completion
echo "All jobs completed at $(date)" 