#!/usr/bin/env python
"""
Benchmark Results Analyzer

This script analyzes and compares benchmark results across different models.
It generates tables and plots to help visualize the performance differences.

Usage:
    python analyze_results.py --log-dir logs/20230101
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import yaml
import json

def find_result_files(log_dir):
    """
    Find all result files in the log directory.
    
    Args:
        log_dir (str): Directory containing benchmark logs
        
    Returns:
        dict: Dictionary mapping model names to result file paths
    """
    result_files = {}
    
    # Look for result files in the log directory
    for model_dir in os.listdir(log_dir):
        model_path = os.path.join(log_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        # Check for task directories
        for task_dir in os.listdir(model_path):
            task_path = os.path.join(model_path, task_dir)
            if not os.path.isdir(task_path):
                continue
            
            # Look for seed directories
            for seed_dir in os.listdir(task_path):
                seed_path = os.path.join(task_path, seed_dir)
                if not os.path.isdir(seed_path) or not seed_dir.startswith("seed"):
                    continue
                
                # Check for results.csv file
                results_file = os.path.join(seed_path, "results.csv")
                if os.path.exists(results_file):
                    if model_dir not in result_files:
                        result_files[model_dir] = []
                    result_files[model_dir].append(results_file)
    
    return result_files

def load_results(result_files):
    """
    Load results from result files.
    
    Args:
        result_files (dict): Dictionary mapping model names to result file paths
        
    Returns:
        pd.DataFrame: DataFrame containing all results
    """
    all_results = []
    
    for model_name, file_paths in result_files.items():
        for file_path in file_paths:
            try:
                # Extract seed from the file path
                seed = os.path.basename(os.path.dirname(file_path)).replace("seed", "")
                
                # Load the results
                df = pd.read_csv(file_path)
                
                # Add model name and seed columns
                df["Model"] = model_name
                df["Seed"] = seed
                
                all_results.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if not all_results:
        print("No results found")
        return None
    
    # Concatenate all results
    return pd.concat(all_results, ignore_index=True)

def analyze_results(results_df):
    """
    Analyze benchmark results.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing all results
        
    Returns:
        dict: Dictionary containing analysis results
    """
    if results_df is None or results_df.empty:
        return {}
    
    # Group by model and calculate statistics
    grouped = results_df.groupby("Model")
    
    # Calculate mean and standard deviation of metrics
    metrics = ["Train metric", "Val metric", "Test metric"]
    stats = {}
    
    for metric in metrics:
        if metric in results_df.columns:
            stats[f"{metric}_mean"] = grouped[metric].mean()
            stats[f"{metric}_std"] = grouped[metric].std()
    
    # Calculate training time statistics
    if "Training time (s)" in results_df.columns:
        stats["Training_time_mean"] = grouped["Training time (s)"].mean()
        stats["Training_time_std"] = grouped["Training time (s)"].std()
    
    # Convert to DataFrame for easier handling
    stats_df = pd.DataFrame(stats)
    
    return stats_df

def plot_results(stats_df, output_dir):
    """
    Generate plots for benchmark results.
    
    Args:
        stats_df (pd.DataFrame): DataFrame containing analysis results
        output_dir (str): Directory to save plots
    """
    if stats_df is None or stats_df.empty:
        print("No data to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot test metrics
    if "Test metric_mean" in stats_df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x=stats_df.index,
            y=stats_df["Test metric_mean"],
            yerr=stats_df["Test metric_std"],
            palette="viridis"
        )
        ax.set_title("Test Metric Comparison", fontsize=16)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel("Mean Absolute Error (Å)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f"{p.get_height():.4f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0,
                xytext=(0, 5),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "test_metric_comparison.png"), dpi=300)
        plt.close()
    
    # Plot training time
    if "Training_time_mean" in stats_df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x=stats_df.index,
            y=stats_df["Training_time_mean"],
            yerr=stats_df["Training_time_std"],
            palette="rocket"
        )
        ax.set_title("Training Time Comparison", fontsize=16)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel("Training Time (seconds)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f"{p.get_height():.1f}s",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0,
                xytext=(0, 5),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), dpi=300)
        plt.close()
    
    # Save the statistics to CSV
    stats_df.to_csv(os.path.join(output_dir, "benchmark_statistics.csv"))
    
    # Create a summary table as HTML
    html_table = stats_df.to_html()
    with open(os.path.join(output_dir, "benchmark_summary.html"), "w") as f:
        f.write("<html><body>\n")
        f.write("<h1>Benchmark Results Summary</h1>\n")
        f.write(html_table)
        f.write("\n</body></html>")

def main():
    """Main function to parse arguments and analyze results."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--log-dir", type=str, required=True, help="Directory containing benchmark logs")
    parser.add_argument("--output-dir", type=str, default="benchmark_analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Find result files
    print(f"Looking for result files in {args.log_dir}")
    result_files = find_result_files(args.log_dir)
    
    if not result_files:
        print("No result files found")
        return
    
    print(f"Found results for {len(result_files)} models")
    
    # Load results
    results_df = load_results(result_files)
    
    if results_df is None or results_df.empty:
        print("No results loaded")
        return
    
    print(f"Loaded {len(results_df)} result entries")
    
    # Analyze results
    stats_df = analyze_results(results_df)
    
    if stats_df.empty:
        print("No statistics calculated")
        return
    
    # Print statistics
    print("\nBenchmark Statistics:")
    print(stats_df)
    
    # Plot results
    plot_results(stats_df, args.output_dir)
    print(f"Analysis results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 