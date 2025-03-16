#!/bin/bash
# Script for submitting benchmark jobs to a SLURM queue system

# Default values
CONFIG_DIR="model_configs"
USE_CPU=false
MEMORY="16G"
TIME="12:00:00"
PARTITION="gpu"  # Change to your server's partition name

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpu)
      USE_CPU=true
      PARTITION="cpu"  # Change to CPU partition if using CPU
      shift
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --memory)
      MEMORY="$2"
      shift 2
      ;;
    --time)
      TIME="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# First, generate the configuration files if they don't exist
if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls -A $CONFIG_DIR)" ]; then
  echo "Generating configuration files..."
  python run_server_benchmarks.py --generate-configs --config-dir "$CONFIG_DIR"
fi

# Get all configuration files
CONFIG_FILES=$(ls ${CONFIG_DIR}/*_config.yaml)

# Submit a job for each configuration file
for CONFIG_FILE in $CONFIG_FILES; do
  # Extract model name from the file name
  MODEL_NAME=$(basename $CONFIG_FILE _config.yaml | tr '[:lower:]' '[:upper:]')
  
  # Create a job script
  JOB_SCRIPT="job_${MODEL_NAME}.sh"
  
  echo "Creating job script for $MODEL_NAME: $JOB_SCRIPT"
  
  # Write the job script
  cat > $JOB_SCRIPT << EOL
#!/bin/bash
#SBATCH --job-name=bench_${MODEL_NAME}
#SBATCH --output=bench_${MODEL_NAME}_%j.out
#SBATCH --error=bench_${MODEL_NAME}_%j.err
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --partition=${PARTITION}
EOL

  # Add GPU resources if not using CPU
  if [ "$USE_CPU" = false ]; then
    cat >> $JOB_SCRIPT << EOL
#SBATCH --gres=gpu:1
EOL
  fi

  # Add the command to run the benchmark
  cat >> $JOB_SCRIPT << EOL

# Load necessary modules (adjust for your environment)
module load python/3.8
module load cuda/11.3  # Only if using GPU

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the benchmark
echo "Running benchmark for ${MODEL_NAME}"
python run_server_benchmarks.py --model ${MODEL_NAME} --config-dir ${CONFIG_DIR} $([ "$USE_CPU" = true ] && echo "--cpu")

echo "Benchmark completed"
EOL

  # Make the job script executable
  chmod +x $JOB_SCRIPT
  
  # Submit the job
  echo "Submitting job for $MODEL_NAME"
  sbatch $JOB_SCRIPT
  
  # Wait a bit between submissions
  sleep 1
done

echo "All jobs submitted" 