import os
import yaml
import torch
import argparse
import time
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter

from CHILI_centralAtoms import CHILI
from vector_diff import VectorDiffusion
from reporter import Reporter
from benchmark_tasks_utils import pos_abs_padded, position_MAE, pos_abs_from_xPDF

def create_small_dataset(dataset, num_samples=5):
    """Create a small subset of the dataset with just a few samples."""
    # Get all indices
    all_indices = list(range(len(dataset)))
    
    # Select first num_samples indices
    selected_indices = all_indices[:num_samples]
    
    # Create a new dataset with just these samples
    small_dataset = torch.utils.data.Subset(dataset, selected_indices)
    
    return small_dataset

def validate_dataset_atom_count(dataset, max_atoms, split_name):
    """Validate that all structures have fewer atoms than max_atoms."""
    for i, data in enumerate(dataset):
        num_atoms = data.pos_abs.shape[0]
        if num_atoms > max_atoms:
            raise ValueError(f"Structure {i} in {split_name} set has {num_atoms} atoms, which exceeds the maximum of {max_atoms}")
    return True

def run_benchmark_overfitting(config_path=None, use_cpu=False):
    """Run a benchmark with a 5-sample dataset to test model overfitting capability."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    if config_path:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        # Default configuration for overfitting test
        config = {
            "root": "./data",
            "dataset": "CHILI-3K",
            "graph_type": "central",
            "task": "AbsPositionRegressionxPDF",
            "model": "VectorDiffusion",
            "log_dir": "./logs/overfitting_test",
            "save_latest_model": True,
            "custom_loss": True,
            "use_reporter": True,
            
            "Reporter_config": {
                "visualization_period": 25,  # Visualize every 25 epochs
                "timesteps_to_visualize": [0, 25, 50, 75, 100],
                "num_structures": 1,
                "figsize": [15, 8]
            },
            
            "Model_config": {
                "in_channels": 6000,
                "hidden_channels": 40,
                "out_channels": 300,
                "T": 100,
                "beta_1": 0.0001,
                "beta_T": 0.02
            },
            
            # Training configuration
            "Train_config": {
                "batch_size": 2,  # Minimum batch size for BatchNorm
                "learning_rate": 0.0004,  # Slightly reduced learning rate for more stable training
                "lr_scheduler": True,  # Enable learning rate scheduler
                "lr_step_size": 75,  # Less frequent learning rate changes
                "lr_gamma": 0.7,  # Less aggressive reduction
                "epochs": 750,  # Extended epoch count to see full potential
                "train_time": 172800,  # 48 hours in seconds - double the time
                "max_patience": 9999,  # Effectively disable early stopping
                "seeds": [42],  # Single seed for one thorough run
                "weight_decay": 0.00005  # Reduced weight decay for less regularization 
            }
        }
    
    # Create dataset
    dataset = CHILI(root=config["root"], dataset=config["dataset"], graph_type=config["graph_type"])
    
    # Create small dataset with just 5 samples
    small_dataset = create_small_dataset(dataset)
    print(f"Created small dataset with {len(small_dataset)} samples")
    
    # Calculate maximum atoms the model can handle
    max_atoms = config["Model_config"]["out_channels"] // 3
    
    # Validate that all structures have fewer atoms than max_atoms
    print(f"Validating dataset (max atoms: {max_atoms})...")
    validate_dataset_atom_count(small_dataset, max_atoms, "small dataset")
    print("Dataset validation successful - all structures have appropriate atom counts.")
    
    # Split into train and validation sets (4 train, 1 validation)
    train_indices = list(range(4))
    val_indices = [4]
    train_dataset = torch.utils.data.Subset(small_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(small_dataset, val_indices)
    
    print(f"Dataset: {config['dataset']} (Small subset for overfitting test)")
    print(f"Task: {config['task']}")
    print(f"Model: {config['model']}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["Train_config"]["batch_size"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["Train_config"]["batch_size"],
        shuffle=False
    )
    
    # Create results dataframe
    results_df = pd.DataFrame(
        columns=[
            "Model",
            "Dataset",
            "Task",
            "Seed",
            "Train samples",
            "Val samples",
            "Train time",
            "Trainable parameters",
            "Train loss",
            "Metric name",
            "Val metric",
        ]
    )

    # Define model configurations
    model_configurations = {
        "VectorDiffusion": {
            "class": VectorDiffusion,
            "kwargs": {"x": "data.y['xPDF']", "batch": "data.batch"},
            "skip_training": False,
        },
    }
        # Import task functions from local benchmarking module
    from benchmark_tasks_utils import (
        pos_abs_from_saxs,
        pos_abs_from_xrd,
        pos_abs_from_xPDF,
        position_MAE,

    )

    # Define task configurations
    task_configurations = {
        "AbsPositionRegressionSAXS": {
            "task_function": pos_abs_from_saxs,
            "loss_function": torch.nn.SmoothL1Loss(),
            "metric_function": position_MAE,
            "metric_name": 'PositionMAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "AbsPositionRegressionXRD": {
            "task_function": pos_abs_from_xrd,
            "loss_function": torch.nn.SmoothL1Loss(),
            "metric_function": position_MAE,
            "metric_name": 'PositionMAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "AbsPositionRegressionxPDF": {
            "task_function": pos_abs_from_xPDF,
            "loss_function": torch.nn.SmoothL1Loss(),
            "metric_function": position_MAE,
            "metric_name": 'PositionMAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
    }
    # Run benchmarks for each seed
    for seed_idx, seed in enumerate(config['Train_config']['seeds']):
        seed_everything(seed)
        print(f"\nRunning with seed: {seed}")
        
        # Get model configuration
        model_config = model_configurations.get(config['model'])
        if model_config is None:
            raise ValueError(f"Model {config['model']} not supported")
        
        model_class = model_config['class']
        model_kwargs = model_config['kwargs']
        model = model_class(**config['Model_config']).to(device)
        
        # Get task configuration
        task_config = task_configurations.get(config['task'])
        if task_config is None:
            raise ValueError(f"Task {config['task']} not implemented")
        
        task_function = task_config['task_function']
        loss_function = task_config['loss_function']
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["Train_config"]["learning_rate"],
            weight_decay=config["Train_config"].get("weight_decay", 0)
        )
        
        # Setup learning rate scheduler if enabled
        if config["Train_config"].get("lr_scheduler", False):
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["Train_config"].get("lr_step_size", 50),
                gamma=config["Train_config"].get("lr_gamma", 0.5)
            )
        else:
            scheduler = None
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_count_str = f"{trainable_params/1000:.0f}k" if trainable_params >= 1000 else str(trainable_params)
        
        # Print model summary
        print(f"\nModel Summary for {config['model']}:")
        print(f"Total trainable parameters: {trainable_params:,} ({param_count_str})")
        
        # Print detailed model architecture
        print("Model architecture:")
        print(model)
        
        # Create descriptive model folder name with parameters
        model_folder_name = f"{config['model']}_{param_count_str}"
        
        # Add key hyperparameters to folder name if they exist
        if 'hidden_channels' in config['Model_config']:
            model_folder_name += f"_h{config['Model_config']['hidden_channels']}"
        if 'T' in config['Model_config'] and config['model'] == 'VectorDiffusion':
            model_folder_name += f"_T{config['Model_config']['T']}"
            
        # Setup logging directory with model name and parameters
        save_dir = f"./logs/overfitting_test/{model_folder_name}/seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(save_dir)
        
        # Initialize reporter for visualization
        reporter = Reporter(config, device, save_dir) if config.get("use_reporter", False) else None
        
        # Training setup
        max_training_time = config["Train_config"]["train_time"]
        start_time = time.time()
        max_patience = config["Train_config"]["max_patience"]
        patience = 0
        best_error = None
        sample_batch = None
        
        # Training loop
        for epoch in range(config["Train_config"]["epochs"]):
            # Check if time limit is reached
            if time.time() - start_time > config["Train_config"]["train_time"]:
                print("Time limit reached")
                break

            # Check patience
            if patience >= max_patience:
                print("Maximum patience reached")
                break
            
            
            # Training
            model.train()
            train_loss = 0
            visualization_done = False
            
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                
                # Forward pass - use standard task function and loss for all models
                pred, truth = task_function(data, model, None, model_kwargs, device, config)
                loss = loss_function(pred, truth)
                
                # Store a sample batch for visualization if needed
                if sample_batch is None and config['model'] == "VectorDiffusion":
                    # Get the input data for visualization
                    xPDF = eval(model_kwargs["x"]) if model_kwargs["x"] != "None" else None
                    sct = xPDF[:,1,:]
                    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                    sct = (sct - sct_min) / (sct_max - sct_min)
                    
                    sample_positions = truth[0].clone().unsqueeze(0)
                    sample_xPDF = sct[0].clone().unsqueeze(0)
                
                # Run reporter for visualization if needed
                if reporter is not None and not visualization_done and reporter.should_visualize(epoch) and batch_idx == 0:
                    model.eval()
                    reporter.visualize_diffusion(
                        epoch, 
                        model, 
                        sample_positions,
                        sample_xPDF,
                        diffusion_steps=config.get("Model_config", {}).get("T", 100)
                    )
                    visualization_done = True
                    model.train()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_error = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    
                    # Use task function to get predictions and ground truth
                    pred, truth = pos_abs_from_xPDF(data, model, None, {"x": "data.y['xPDF']", "batch": "data.batch"}, device, config)
                    metric = position_MAE(pred, truth)
                    val_error += metric.item()
            
            val_error /= len(val_loader)
            
            # Run reporter for visualization if needed
            if reporter is not None and reporter.should_visualize(epoch):
                model.eval()
                reporter.visualize_diffusion(
                    epoch, 
                    model, 
                    sample_positions,
                    sample_xPDF,
                    diffusion_steps=config["Model_config"]["T"]
                )
                model.train()
            
            # Log metrics
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("PositionMAE/val", val_error, epoch)
            
            # Step the learning rate scheduler if it exists
            if scheduler is not None:
                scheduler.step()
                writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
            
            print(f"Epoch: {epoch+1}/{config['Train_config']['epochs']}, Train Loss: {train_loss:.4f}, Val PositionMAE: {val_error:.4f}")
            
            # Save model if improved
            if best_error is None or val_error < best_error:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                    },
                    f"{save_dir}/best.pt"
                )
                best_error = val_error
                patience = 0
            else:
                patience += 1
            
            # Save latest model
            if config.get("save_latest_model", False):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                    },
                    f"{save_dir}/latest.pt"
                )
            
            # Check patience
            if patience >= config["Train_config"].get("max_patience", 9999):
                print("Maximum patience reached")
                break
        
        # Training complete
        stop_time = time.time()
        
        # Close writer
        writer.close()
        
        # Print final results
        print(f"\nTraining completed in {stop_time - start_time:.2f} seconds")
        print(f"Best validation PositionMAE: {best_error:.4f}")
        print(f"Results saved to {save_dir}")
        
        # Add results to dataframe
        results_df.loc[seed_idx] = [
            config["model"],
            f"{config['dataset']} (5 samples)",
            config["task"],
            seed,
            len(train_dataset),
            len(val_dataset),
            stop_time - start_time,
            trainable_params,
            train_loss,
            "PositionMAE",
            best_error,
        ]
    
    # Save results with model name and parameters
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f"{results_dir}/overfitting_results.csv")
    print(f"Results saved to {results_dir}/overfitting_results.csv")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a benchmark with minimal data to test overfitting capabilities')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()
    
    run_benchmark_overfitting(args.config, args.cpu)
