#!/usr/bin/env python
"""
Server Benchmark Runner

This script runs benchmarks for multiple models using separate YAML files for each model.
It's designed to be run on a server environment and supports:
- Running all models in sequence
- Running a specific model by name
- Generating model-specific YAML files from a template

Usage:
    python run_server_benchmarks.py --all                # Run all model benchmarks
    python run_server_benchmarks.py --model MLP          # Run a specific model
    python run_server_benchmarks.py --generate-configs   # Generate config files for all models
"""

import os
import sys
import yaml
import argparse
import glob
import time
from datetime import datetime
import subprocess
import copy

# Import the benchmark function
from benchmark import run_benchmark_from_notebook

# Define available models
AVAILABLE_MODELS = [
    "MLP", 
    "GCN", 
    "GraphSAGE", 
    "GIN", 
    "GAT", 
    "EdgeCNN"
]

def generate_model_configs(template_path="example_config.yaml", output_dir="model_configs"):
    """
    Generate separate YAML configuration files for each model based on a template.
    
    Args:
        template_path (str): Path to the template configuration file
        output_dir (str): Directory to save the generated configuration files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the template configuration
    with open(template_path, "r") as file:
        template_config = yaml.safe_load(file)
    
    # Generate a configuration file for each model
    for model_name in AVAILABLE_MODELS:
        # Create a copy of the template and update the model
        config = copy.deepcopy(template_config)
        config["model"] = model_name
        
        # Add a timestamp to the log directory to avoid conflicts
        config["log_dir"] = os.path.join(config["log_dir"], f"{datetime.now().strftime('%Y%m%d')}")
        
        # Save the configuration to a file
        config_path = os.path.join(output_dir, f"{model_name.lower()}_config.yaml")
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Generated configuration file for {model_name}: {config_path}")

def run_benchmark(config_path, use_cpu=False, debug_mode=False):
    """
    Run a benchmark using the specified configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        use_cpu (bool): Whether to use CPU instead of GPU
        debug_mode (bool): Whether to enable CUDA debug mode
    
    Returns:
        dict: Results of the benchmark
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark with configuration: {config_path}")
    print(f"{'='*60}")
    
    # Enable CUDA debug mode if requested
    if debug_mode and not use_cpu:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("CUDA debug mode enabled (CUDA_LAUNCH_BLOCKING=1)")
    
    try:
        # Run the benchmark
        start_time = time.time()
        results = run_benchmark_from_notebook(config_path, use_cpu)
        elapsed_time = time.time() - start_time
        
        # Print summary
        print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")
        print(f"Results: {results}")
        
        return results
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return {"error": str(e)}

def run_all_benchmarks(config_dir="model_configs", use_cpu=False, debug_mode=False):
    """
    Run benchmarks for all models using their respective configuration files.
    
    Args:
        config_dir (str): Directory containing the configuration files
        use_cpu (bool): Whether to use CPU instead of GPU
        debug_mode (bool): Whether to enable CUDA debug mode
    
    Returns:
        dict: Dictionary of results for each model
    """
    # Find all configuration files
    config_files = glob.glob(os.path.join(config_dir, "*_config.yaml"))
    
    if not config_files:
        print(f"No configuration files found in {config_dir}")
        print("Run with --generate-configs to create configuration files")
        return {}
    
    results = {}
    
    # Run benchmarks for each configuration file
    for config_path in sorted(config_files):
        # Extract model name from the file name
        model_name = os.path.basename(config_path).split("_")[0].upper()
        
        # Run the benchmark
        model_results = run_benchmark(config_path, use_cpu, debug_mode)
        results[model_name] = model_results
    
    return results

def main():
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Run benchmarks for multiple models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all model benchmarks")
    group.add_argument("--model", type=str, help="Run a specific model benchmark")
    group.add_argument("--generate-configs", action="store_true", help="Generate configuration files for all models")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--debug", action="store_true", help="Enable CUDA debug mode (sets CUDA_LAUNCH_BLOCKING=1)")
    parser.add_argument("--config-dir", type=str, default="model_configs", help="Directory containing model configurations")
    
    args = parser.parse_args()
    
    # Generate configuration files if requested
    if args.generate_configs:
        generate_model_configs(output_dir=args.config_dir)
        return
    
    # Run a specific model benchmark
    if args.model:
        model_name = args.model.upper()
        if model_name not in AVAILABLE_MODELS:
            print(f"Error: Model {model_name} not in available models: {', '.join(AVAILABLE_MODELS)}")
            return
        
        config_path = os.path.join(args.config_dir, f"{model_name.lower()}_config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: Configuration file {config_path} not found")
            print("Run with --generate-configs to create configuration files")
            return
        
        run_benchmark(config_path, args.cpu, args.debug)
        return
    
    # Run all model benchmarks
    if args.all:
        results = run_all_benchmarks(config_dir=args.config_dir, use_cpu=args.cpu, debug_mode=args.debug)
        
        # Print summary of all results
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        for model_name, result in results.items():
            if "error" in result:
                print(f"{model_name}: Error - {result['error']}")
            else:
                print(f"{model_name}: Success")
        
        return

if __name__ == "__main__":
    main() 