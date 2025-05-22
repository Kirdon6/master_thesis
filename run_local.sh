#!/bin/bash
# Simple script to run diffusion experiments locally for testing

# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/diffusion_xpdf.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "No GPU detected, will run on CPU"
fi

# Run the diffusion script
echo "Running diffusion experiment with config: $CONFIG_FILE"
echo "Output will be saved to logs/diffusion.log"
python run_diffusion.py --config_path "$CONFIG_FILE" 2>&1 | tee logs/diffusion.log

echo "Experiment completed at $(date)" 