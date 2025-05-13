#!/usr/bin/env python
"""
SLURM-optimized script for running structure prediction experiments on CHILI dataset.
This script can use either the diffusion model from mnist_ddpm_cond.py or the baseline
MLP model from baseline_model.py to predict structures from xPDF or XRD data.

Usage:
    python run_diffusion.py --config_path configs/diffusion_xpdf.yaml
    python run_diffusion.py --config_path configs/mlp_xpdf.yaml

When running on SLURM:
    sbatch submit_diffusion.sh
"""

import os
import sys
import yaml
import argparse
import time
import datetime
import logging
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import torch.nn as nn



# Import the diffusion model and related functions
from mnist_ddpm_cond import train_vector_conditioned_ddpm
from baseline_model import train_mlp_model
from CHILI_centralAtoms import CHILI

# Set up logging
def setup_logging(log_dir, job_id=None, task_id=None):
    """Set up logging to file and console."""
    if job_id is None:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
    if task_id is None:
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    # Extract model type from config file name or environment variable
    model_type = os.environ.get('MODEL_TYPE', 'unknown')
    log_file = f"{log_dir}/{model_type}_job{job_id}_task{task_id}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_gpu_environment():
    """Configure GPU environment based on SLURM assignment."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        return "cpu"
    
    # If SLURM assigns a specific GPU, use that
    slurm_gpu = os.environ.get('SLURM_JOB_GPUS')
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if slurm_gpu:
        logging.info(f"SLURM assigned GPU(s): {slurm_gpu}")
        # SLURM already sets CUDA_VISIBLE_DEVICES, but we can ensure it:
        if not cuda_visible_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = slurm_gpu
    elif cuda_visible_devices:
        logging.info(f"Using GPU(s) from CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    else:
        # No specific GPU assignment, use all available
        logging.info(f"No specific GPU assignment, using all {torch.cuda.device_count()} available GPUs")
    
    # Log GPU info
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return "cuda"

def setup_model(model, use_multi_gpu=False):
    """Prepare model for single or multi-GPU training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        logging.info(f"Found {n_gpus} CUDA device(s)")
        
        if use_multi_gpu and n_gpus > 1:
            logging.info(f"Using {n_gpus} GPUs with DataParallel")
            model = nn.DataParallel(model)
            
    model = model.to(device)
    return model, device

def run_experiment(config_path, use_multi_gpu=False):
    """
    Run a structure prediction experiment using either diffusion or MLP model based on configuration.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    use_multi_gpu : bool, optional
        Whether to use multiple GPUs if available
    """
    # Set up SLURM environment variables
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create log directory
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_dir, job_id, task_id)
    
    # Determine model type from config
    model_type = config.get('model', 'Diffusion')
    task_type = config.get('task', 'unknown')
    logger.info(f"Starting experiment with model: {model_type}")
    logger.info(f"SLURM Job ID: {job_id}, Task ID: {task_id}")
    logger.info(f"Task type: {task_type}")
    
    # Set up GPU environment
    device_type = setup_gpu_environment()
    device = torch.device(device_type)
    logger.info(f"Using device: {device}")
    
    # Check for multi-GPU support
    if use_multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logger.info(f"Multi-GPU mode enabled with {num_gpus} GPUs")
        else:
            logger.info(f"Multi-GPU mode requested but only {num_gpus} GPU available")
    
    # Set random seed for reproducibility
    # Get seed from config based on model type
    seed = config.get('Train_config', {}).get('seed', 42)
    
    seed_everything(seed)
    logger.info(f"Using random seed: {seed}")
    
    # Load dataset
    dataset_config = config.get('dataset', {})
    dataset_root = dataset_config.get('root', config.get('root', 'data'))
    dataset_name = dataset_config.get('name', config.get('dataset', 'CHILI-3K'))
    graph_type = dataset_config.get('graph_type', config.get('graph_type', 'central'))
    
    logger.info(f"Loading dataset from {dataset_root}/{dataset_name}")
    dataset = CHILI(
        root=dataset_root,
        dataset=dataset_name,
        graph_type=graph_type
    )
    
    # Load data split
    try:
        dataset.load_data_split(split_strategy='random')
    except FileNotFoundError:
        dataset.create_data_split(split_strategy = 'random', test_size=0.1)


    num_workers = config.get('Train_config', {}).get('num_workers', 1)
    
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=config['Train_config'].get('batch_size', 64),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset.validation_set,
        batch_size=config['Train_config'].get('batch_size', 64),
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset.test_set,
        batch_size=config['Train_config'].get('batch_size', 64),
        shuffle=False,
        num_workers=num_workers,
    )
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Train samples: {len(dataset.train_set)}")
    logger.info(f"Validation samples: {len(dataset.validation_set)}")
    logger.info(f"Test samples: {len(dataset.test_set)}")
    
    # Set the output directory with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    task_type = config.get('task', 'unknown')
    output_dir = os.path.join(
        config.get("output_dir", "diffusion_results"),
        f"{task_type}_{timestamp}_job{job_id}_task{task_id}"
    )
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Save the config file in the output directory
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Start the appropriate training based on model type
    start_time = time.time()
    try:
        if model_type.lower() == 'mlp':
            logger.info("Starting MLP baseline model training...")

            model_params = {
                "in_channels": config['Model_config'].get('in_channels', 6000),
                "hidden_dim": config['Model_config'].get('hidden_dim', 512),
                "num_layers": config['Model_config'].get('num_layers', 3),
                "dropout": config['Model_config'].get('dropout', 0.1),
                "batch_size": config['Train_config'].get('batch_size', 64),
                "learning_rate": config['Train_config'].get('learning_rate', 1e-3),
                "weight_decay": config['Train_config'].get('weight_decay', 0.0001),
                "epochs": config['Train_config'].get('epochs', 100),                
                "max_patience": config['Train_config'].get('max_patience', 20),
                "lr_scheduler": config['Train_config'].get('lr_scheduler', True),
                "lr_step_size": config['Train_config'].get('lr_step_size', 30),
                "lr_gamma": config['Train_config'].get('lr_gamma', 0.1),
                 "model_type": config['Model_config'].get('model_type', 'pos_frac'),               
            }

            # Add atom_mapping_path if it exists
            atom_mapping_path = config['Model_config'].get('atom_mapping_path', None)
            if atom_mapping_path:
                if os.path.exists(atom_mapping_path):
                    model_params["atom_mapping_path"] = atom_mapping_path
                    logger.info(f"Using atom mapping from {atom_mapping_path}")
            else:
                logger.warning(f"Atom mapping file {atom_mapping_path} not found. Proceeding without atom types.")

            # Continue with MLP training
            model, metrics = train_mlp_model(
                train_loader,
                val_loader,
                test_loader,
                sample_dir=os.path.join(output_dir, "samples"),
                cond_type=config['Model_config'].get('cond_type', 'xPDF'),
                **model_params
            )
        else:
            logger.info("Starting diffusion model training...")
            # Configure the Diffusion model parameters from the config
            model_params = {
                "T": config['Model_config'].get('T', 1000),
                "learning_rate": config['Train_config'].get('learning_rate', 1e-3),
                "epochs": config['Train_config'].get('epochs', 100),
                "batch_size": config['Train_config'].get('batch_size', 64),
                "ema": config['Train_config'].get('ema', True),
                "cond_dim": config['Model_config'].get('cond_dim', 6000),
                "cond_embed_dim": config['Model_config'].get('cond_embed_dim', 64),
                "image_size": tuple(config['Model_config'].get('image_size', (10, 10))),
                "model_type": config['Model_config'].get('model_type', 'pos_frac'),
            }
            
            # Add atom_mapping_path if it exists
            atom_mapping_path = config['Model_config'].get('atom_mapping_path', None)
            if atom_mapping_path:
                if os.path.exists(atom_mapping_path):
                    model_params["atom_mapping_path"] = atom_mapping_path
                    logger.info(f"Using atom mapping from {atom_mapping_path}")
            else:
                logger.warning(f"Atom mapping file {atom_mapping_path} not found. Proceeding without atom types.")
            
            # Add this inside the diffusion training section, before calling train_vector_conditioned_ddpm
            reporter_freq = config['Train_config'].get('reporter_save_frequency', 10)
            logger.info(f"Setting reporter to save images every {reporter_freq} epochs")
            
            
            # Train diffusion model
            model, metrics = train_vector_conditioned_ddpm(
                train_data=train_loader,
                val_data=val_loader,
                test_data=test_loader,
                sample_dir=os.path.join(output_dir, "samples"),
                cond_type=config['Model_config'].get('cond_type', 'xPDF'),
                **model_params
            )
        
        # Calculate and log total training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Log final metrics
        if model_type.lower() == 'mlp':
            logger.info(f"Final test MAE: {metrics['test_mae']:.4f}")
            logger.info(f"Final test Hausdorff: {metrics['test_hausdorff']:.4f}")
        else:
            if 'val_maes' in metrics and len(metrics['val_maes']) > 0:
                logger.info(f"Final validation MAE: {metrics['val_maes'][-1]:.4f}")
            if 'test_mae' in metrics:
                logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
            if 'test_hausdorff' in metrics:
                logger.info(f"Test Hausdorff: {metrics['test_hausdorff']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Experiment completed successfully!")
    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run structure prediction experiments with diffusion or MLP models")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    run_experiment(
        args.config_path
    ) 