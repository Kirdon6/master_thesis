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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the diffusion model and related functions
from mnist_ddpm_cond import (
    train_vector_conditioned_ddpm)
# Import the MLP baseline model
from baseline_model import BaselineMLP
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

def train_mlp_model(config, train_loader, val_loader, test_loader, device, output_dir):
    """
    Train and evaluate the MLP baseline model.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run training on
    output_dir : str
        Directory to save model and results
        
    Returns
    -------
    model : BaselineMLP
        Trained MLP model
    metrics : dict
        Dictionary of training and evaluation metrics
    """
    # Get model configuration
    model_config = config.get('Model_config', {})
    train_config = config.get('Train_config', {})
    
    # Create the MLP model
    model = BaselineMLP(
        in_channels=model_config.get('in_channels', 6000),
        hidden_channels=model_config.get('hidden_dim', 512),
        num_layers=model_config.get('num_layers', 3),
        dropout=model_config.get('dropout', 0.1)
    ).to(device)
    
    logging.info(f"Created BaselineMLP with {model_config.get('num_layers', 3)} layers")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.get('learning_rate', 0.001),
        weight_decay=train_config.get('weight_decay', 0.0001)
    )
    
    # Setup learning rate scheduler
    if train_config.get('lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.get('lr_step_size', 30),
            gamma=train_config.get('lr_gamma', 0.5)
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)  # No-op scheduler
    
    # Setup early stopping
    best_val_mae = float('inf')
    patience = train_config.get('max_patience', 20)
    patience_counter = 0
    best_epoch = 0
    
    # Metrics to track
    train_losses = []
    val_losses = []
    val_maes = []
    val_hausdorffs = []  # Add tracking for validation Hausdorff distances
    
    # Setup loss function
    criterion = torch.nn.SmoothL1Loss()
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Function to visualize ground truth vs predictions
    def visualize_predictions(epoch):
        """Create a visualization of ground truth vs predicted structures"""
        model.eval()
        
        with torch.no_grad():
            # Get 2 samples from validation set
            val_iter = iter(val_loader)
            batch = next(val_iter)
            
            # Extract xPDF data and positions
            if config.get('task', '').lower() == 'abspositionregressionxpdf':
                xpdf = batch.y['xPDF']
                # Normalize xPDF data
                sct = xpdf[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_abs.reshape(-1, 100, 3)
            elif config.get('task', '').lower() == 'abspositionregressionxrd':
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min) 
                y = batch.pos_abs.view(-1, 100, 3) 
            elif config.get('task', '').lower() == 'fracpositionregressionxpdf':
                xpdf = batch.y['xPDF']
                sct = xpdf[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_frac.view(-1, 100, 3)
            elif config.get('task', '').lower() == 'fracpositionregressionxrd':
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_frac.view(-1, 100, 3)
            
            # Take only the first 2 samples
            sct = sct[:2].to(device)
            y = y[:2].to(device)
            
            # Get model predictions
            predictions = model(sct)
            
            # Create 2x2 grid for visualization
            fig = plt.figure(figsize=(15, 15), facecolor='white')
            
            # Style function for 3D plots
            def style_3d_axes(ax, title):
                ax.set_facecolor('white')
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('lightgray')
                ax.yaxis.pane.set_edgecolor('lightgray')
                ax.zaxis.pane.set_edgecolor('lightgray')
                ax.set_title(title, fontsize=14, pad=10)
                ax.set_xlabel('X', fontsize=10, labelpad=5)
                ax.set_ylabel('Y', fontsize=10, labelpad=5)
                ax.set_zlabel('Z', fontsize=10, labelpad=5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            
            # Plot ground truth and predictions
            for i in range(2):
                # Ground truth
                ax_gt = fig.add_subplot(2, 2, i*2+1, projection='3d')
                gt_points = y[i].cpu().numpy()
                ax_gt.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                            c='blue', marker='o', s=25, alpha=0.8)
                style_3d_axes(ax_gt, f'Ground Truth Structure {i+1}')
                
                # Prediction
                ax_pred = fig.add_subplot(2, 2, i*2+2, projection='3d')
                pred_points = predictions[i].cpu().numpy()
                ax_pred.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                                c='red', marker='o', s=25, alpha=0.8)
                style_3d_axes(ax_pred, f'MLP Prediction {i+1}')
                
                # Calculate and display MAE
                from benchmark_tasks_utils import position_MAE
                mae = position_MAE(predictions[i:i+1], y[i:i+1]).item()
                
                # Calculate and display Hausdorff distance
                from scipy.spatial.distance import directed_hausdorff
                fwd_hausdorff = directed_hausdorff(pred_points, gt_points)[0]
                bwd_hausdorff = directed_hausdorff(gt_points, pred_points)[0]
                hausdorff = max(fwd_hausdorff, bwd_hausdorff)
                
                # Display metrics
                ax_pred.text2D(0.05, 0.95, f'MAE: {mae:.4f}', transform=ax_pred.transAxes, fontsize=12)
                ax_pred.text2D(0.05, 0.90, f'Hausdorff: {hausdorff:.4f}', transform=ax_pred.transAxes, fontsize=12)
            
            # plt.tight_layout()
            
            # Save visualization
            filename = os.path.join(viz_dir, f"epoch_{epoch:03d}_structures.png")
            plt.savefig(filename, dpi=200)
            plt.close()
            
            # logging.info(f"Saved structure visualization to {filename}")
    
    # Training loop
    max_epochs = train_config.get('epochs', 100)
    logging.info(f"Starting MLP training for {max_epochs} epochs")
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            # Extract xPDF data and positions
            if config.get('task', '').lower() == 'abspositionregressionxpdf':
                xpdf = batch.y['xPDF']
                # Normalize xPDF data
                sct = xpdf[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_abs.reshape(-1, 100,3)
            elif config.get('task', '').lower() == 'abspositionregressionxrd':
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min) 
                y = batch.pos_abs.view(-1, 100,3) 
            elif config.get('task', '').lower() == 'fracpositionregressionxpdf':
                xpdf = batch.y['xPDF']
                sct = xpdf[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_frac.view(-1, 100,3)
            elif config.get('task', '').lower() == 'fracpositionregressionxrd':
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_frac.view(-1, 100,3)
            
            # Move data to device
            sct = sct.to(device)
            y = y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(sct)
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_epoch_losses = []
        all_preds = []
        all_truths = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Extract xPDF data and positions
                if config.get('task', '').lower() == 'abspositionregressionxpdf':
                    xpdf = batch.y['xPDF']
                    # Normalize xPDF data
                    sct = xpdf[:,1,:]
                    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                    sct = (sct - sct_min) / (sct_max - sct_min)
                    y = batch.pos_abs.reshape(-1, 100,3)
                elif config.get('task', '').lower() == 'abspositionregressionxrd':
                    xrd = batch.y['xrd']
                    sct = xrd[:,1,:]
                    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                    sct = (sct - sct_min) / (sct_max - sct_min) 
                    y = batch.pos_abs.view(-1, 100,3) 
                elif config.get('task', '').lower() == 'fracpositionregressionxpdf':
                    xpdf = batch.y['xPDF']
                    sct = xpdf[:,1,:]
                    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                    sct = (sct - sct_min) / (sct_max - sct_min)
                    y = batch.pos_frac.view(-1, 100,3)
                elif config.get('task', '').lower() == 'fracpositionregressionxrd':
                    xrd = batch.y['xrd']
                    sct = xrd[:,1,:]
                    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                    sct = (sct - sct_min) / (sct_max - sct_min)
                    y = batch.pos_frac.view(-1, 100,3)
                
                # Move data to device
                sct = sct.to(device)
                y = y.to(device)
                
                # Forward pass
                predictions = model(sct)
                loss = criterion(predictions, y)
                
                val_epoch_losses.append(loss.item())
                
                all_preds.append(predictions)
                all_truths.append(y)
        
        # Calculate validation metrics
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # Calculate MAE for validation
        all_preds = torch.cat(all_preds, dim=0)
        all_truths = torch.cat(all_truths, dim=0)
        
        # Use same MAE calculation as in diffusion model
        from benchmark_tasks_utils import position_MAE
        val_mae = position_MAE(all_preds, all_truths).item()
        val_maes.append(val_mae)
        
        # Calculate Hausdorff distance for validation set
        from scipy.spatial.distance import directed_hausdorff
        hausdorff_distances = []
        for i in range(all_preds.shape[0]):
            pred_points = all_preds[i].cpu().numpy()
            true_points = all_truths[i].cpu().numpy()
            
            forward_hausdorff = directed_hausdorff(pred_points, true_points)[0]
            backward_hausdorff = directed_hausdorff(true_points, pred_points)[0]
            
            hausdorff = max(forward_hausdorff, backward_hausdorff)
            hausdorff_distances.append(hausdorff)
        
        val_hausdorff = sum(hausdorff_distances) / len(hausdorff_distances)
        val_hausdorffs.append(val_hausdorff)
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}, "
                     f"Val Loss: {avg_val_loss:.6f}, Val MAE: {val_mae:.6f}, Val Hausdorff: {val_hausdorff:.6f}")
        
        # Generate visualization every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
            visualize_predictions(epoch)
        
        # Check for improvement for early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_epoch = epoch
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, "best_mlp_model.pt"))
            logging.info(f"Saved new best model at epoch {epoch+1} with MAE: {val_mae:.6f}")
            
            # Always generate visualization for best model
            visualize_predictions(epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_mlp_model.pt")))
    
    # Test phase
    model.eval()
    test_losses = []
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Extract data and positions based on task
            if config.get('task', '').lower() == 'abspositionregressionxpdf':
                xpdf = batch.y['xPDF']
                # Normalize xPDF data
                sct = xpdf[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_abs.reshape(-1, 100,3)
            elif config.get('task', '').lower() == 'abspositionregressionxrd':
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min) 
                y = batch.pos_abs.view(-1, 100,3) 
            elif config.get('task', '').lower() == 'fracpositionregressionxpdf':
                xpdf = batch.y['xPDF']
                sct = xpdf[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_frac.view(-1, 100,3)
            elif config.get('task', '').lower() == 'fracpositionregressionxrd':
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                y = batch.pos_frac.view(-1, 100,3)
            
            # Move data to device
            sct = sct.to(device)
            y = y.to(device)
            
            # Forward pass
            predictions = model(sct)
            loss = criterion(predictions, y)
            
            test_losses.append(loss.item())
            
            all_preds.append(predictions)
            all_truths.append(y)
    
    # Calculate test metrics
    avg_test_loss = sum(test_losses) / len(test_losses)
    
    # Calculate MAE for test set
    all_preds = torch.cat(all_preds, dim=0)
    all_truths = torch.cat(all_truths, dim=0)
    
    from benchmark_tasks_utils import position_MAE
    test_mae = position_MAE(all_preds, all_truths).item()
    
    # Calculate Hausdorff distance for test set
    from scipy.spatial.distance import directed_hausdorff
    hausdorff_distances = []
    for i in range(all_preds.shape[0]):
        pred_points = all_preds[i].cpu().numpy()
        true_points = all_truths[i].cpu().numpy()
        
        forward_hausdorff = directed_hausdorff(pred_points, true_points)[0]
        backward_hausdorff = directed_hausdorff(true_points, pred_points)[0]
        
        hausdorff = max(forward_hausdorff, backward_hausdorff)
        hausdorff_distances.append(hausdorff)
    
    test_hausdorff = sum(hausdorff_distances) / len(hausdorff_distances)
    
    # Generate final visualization
    visualize_predictions(max_epochs)
    
    # Log final results
    logging.info(f"Test Loss: {avg_test_loss:.6f}, Test MAE: {test_mae:.6f}, "
                 f"Test Hausdorff: {test_hausdorff:.6f}")
    
    # Save metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'val_hausdorffs': val_hausdorffs,  # Add validation Hausdorff distances
        'test_loss': avg_test_loss,
        'test_mae': test_mae,
        'test_hausdorff': test_hausdorff,
        'best_epoch': best_epoch
    }
    
    # Save metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses + [None] * (len(train_losses) - len(val_losses)),
        'val_mae': val_maes + [None] * (len(train_losses) - len(val_maes)),
        'val_hausdorff': val_hausdorffs + [None] * (len(train_losses) - len(val_hausdorffs))  # Add Hausdorff distances
    })
    metrics_df.to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)
    
    # Save final summary
    summary_df = pd.DataFrame({
        'model_type': ['MLP'],
        'hidden_dim': [model_config.get('hidden_dim', 512)],
        'num_layers': [model_config.get('num_layers', 3)],
        'best_epoch': [best_epoch + 1],
        'best_val_mae': [best_val_mae],
        'test_loss': [avg_test_loss],
        'test_mae': [test_mae],
        'test_hausdorff': [test_hausdorff]
    })
    summary_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    return model, metrics

def run_experiment(config_path, resume_from=None):
    """
    Run a structure prediction experiment using either diffusion or MLP model based on configuration.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    resume_from : str, optional
        Path to a checkpoint to resume training from
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
    
    # Create dataloaders
    # Get batch size based on model type

    batch_size = config.get('Train_config', {}).get('batch_size', 32)
    num_workers = config.get('Train_config', {}).get('num_workers', 4)

    
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device_type == "cuda" else False
    )
    val_loader = DataLoader(
        dataset.validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device_type == "cuda" else False
    )
    test_loader = DataLoader(
        dataset.test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device_type == "cuda" else False
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
            model, metrics = train_mlp_model(
                config,
                train_loader,
                val_loader,
                test_loader,
                device,
                output_dir
            )
        else:
            logger.info("Starting diffusion model training...")
            # Configure the Diffusion model parameters from the config
            model_params = {
                "T": config['Model_config'].get('T', 1000),
                "learning_rate": config['Train_config'].get('learning_rate', 1e-3),
                "epochs": config['Train_config'].get('epochs', 100),
                "batch_size": batch_size,
                "ema": config['Train_config'].get('ema', True),
                "cond_dim": config['Model_config'].get('cond_dim', 6000),
                "cond_embed_dim": config['Model_config'].get('cond_embed_dim', 64),
                "image_size": tuple(config['Model_config'].get('image_size', (10, 10))),
                "model_type": config['Model_config'].get('model_type', 'pos_frac')
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
            
            # # Generate samples using the trained model if requested
            # if config.get('generate_samples', True):
            #     logger.info("Generating samples from trained model...")
            #     # Get a few conditioning vectors from the validation set
            #     sample_loader = DataLoader(
            #         dataset.validation_set,
            #         batch_size=5,  # Just get a few samples
            #         shuffle=True
            #     )
            #     sample_batch = next(iter(sample_loader))
                
            #     # Extract conditioning vectors based on task
            #     if config.get('task', '').lower() == 'abspositionregressionxrd' or 'xrd' in config_path.lower():
            #         cond_vectors = sample_batch.y['XRD'][:,1,:]
            #     else:  # Default to xPDF
            #         cond_vectors = sample_batch.y['xPDF'][:,1,:]
                
            #     # Generate and save samples
            #     sample_dir = os.path.join(output_dir, "samples")
            #     os.makedirs(sample_dir, exist_ok=True)
            #     sample_and_save_images(model, cond_vectors, num_samples=10, save_dir=sample_dir)
            #     logger.info(f"Samples saved to {sample_dir}")
        
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
    parser.add_argument("--resume", type=str, help="Path to a checkpoint to resume training from")
    args = parser.parse_args()
    
    run_experiment(args.config_path, args.resume) 