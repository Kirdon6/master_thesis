#!/bin/bash
#SBATCH --job-name=structure_prediction
#SBATCH --output=slurm_logs/%A_%a.out   # Standard output log - %A is job ID, %a is array task ID
#SBATCH --error=slurm_logs/%A_%a.err    # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Use 4 CPU cores
#SBATCH --mem=32G                       # Memory limit
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH --time=48:00:00                 # Time limit (48 hours)
#SBATCH --array=0                     # Array job with 4 tasks (for different configs)

source ~/master_thesis/nano_diff/bin/activate

# Run the experiment script
python test_torch_cluster.py

deactivate

# Signal completion
echo "Job completed at $(date)" 