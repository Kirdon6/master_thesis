#!/bin/bash
#SBATCH --job-name=structure_prediction
#SBATCH --output=slurm_logs/%j.out       # Standard output log - %j is job ID
#SBATCH --error=slurm_logs/%j.err        # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Use 4 CPU cores
#SBATCH --mem=32G                       # Memory limit
#SBATCH --gres=gpu:1                    # Request 1 GPUs
#SBATCH --time=48:00:00                 # Time limit (48 hours)

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Set CUDA debugging flags
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# # IMPORTANT: Set CUDA_VISIBLE_DEVICES *before* activating Python environment
# # Convert Slurm's GPU IDs into proper CUDA device indices (0,1,2,3)
# if [ -n "$SLURM_JOB_GPUS" ]; then
#     # This is a comma-separated list from Slurm like "0,1,2,3" or "1,2,3,4"
#     echo "Slurm assigned GPUs: $SLURM_JOB_GPUS"
    
#     # Always use sequential device IDs starting from 0, regardless of what Slurm assigns
#     export CUDA_VISIBLE_DEVICES=0
#     echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# fi

# Now activate the environment AFTER setting CUDA_VISIBLE_DEVICES
source ~/master_thesis/nano_diff/bin/activate

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "# CPUs: $SLURM_CPUS_PER_TASK"

# Verify GPUs are available
nvidia-smi

# Print CUDA version and GPU info
nvcc --version

# Add error handling
handle_error() {
    echo "Error occurred in config: $current_config"
    echo "Exit code: $?"
    echo "Check logs for details"
}
trap handle_error ERR

# List of config files to run sequentially
CONFIG_FILES=(
    "configs/diffusion_xpdf_abs_base_long_lex.yaml"
    "configs/diffusion_xpdf_abs_base_long_hier.yaml"
    "configs/diffusion_xpdf_abs_base_long_dist.yaml"
    "configs/diffusion_xpdf_frac_base_long_lex.yaml"
    "configs/diffusion_xpdf_frac_base_long_hier.yaml"
    "configs/diffusion_xpdf_frac_base_long_dist.yaml"
)

# Run each config file sequentially
for config in "${CONFIG_FILES[@]}"; do
    echo "----------------------------------------"
    echo "Starting job with config file: $config"
    echo "----------------------------------------"
    current_config=$config
    
    # Extract model type from config file name
    MODEL_TYPE="Unknown"
    if [[ $config == *"diffusion"* ]]; then
        MODEL_TYPE="Diffusion"
    elif [[ $config == *"mlp"* ]]; then
        MODEL_TYPE="MLP"
    fi

    echo "Running experiment with model type: $MODEL_TYPE"

    # Run the experiment script with error checking
    if python run_benchmarks.py --config "$config"; then
        echo "----------------------------------------"
        echo "Completed job with config file: $config"
        echo "----------------------------------------"
        echo ""
    else
        echo "----------------------------------------"
        echo "Failed to run: $config"
        echo "----------------------------------------"
        echo ""
    fi
    
    # Optional: Add a small delay between runs
    sleep 5
done

deactivate

# Signal completion
echo "All jobs completed at $(date)" 
