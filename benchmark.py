import os
import yaml
import argparse
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
from torch_geometric.nn.models import MLP, GCN, GIN, GAT, EdgeCNN, GraphSAGE, GraphUNet

from CHILI_centralAtoms import CHILI
from baseline_model import BaselineMLP

def validate_dataset_atom_count(dataset, max_atoms, split_name):
    """Validate that all structures in the dataset have fewer atoms than max_atoms."""
    for idx in dataset.indices:
        data = dataset.dataset[idx]
        if len(data.pos_abs) > max_atoms:
            raise ValueError(
                f"Structure at index {idx} in {split_name} set has {len(data.pos_abs)} atoms, "
                f"which exceeds the maximum of {max_atoms} atoms supported by the model. "
                f"Increase Model_config.out_channels or use a dataset with smaller structures."
            )
    return True

def run_benchmark(args):
    # Load configuration
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = CHILI(root=config["root"], dataset=config["dataset"], graph_type=config["graph_type"])
    
    # Load or create data split
    try:
        dataset.load_data_split(split_strategy = 'random')
    except FileNotFoundError:
        print("No data split found, first run create_data_split")
    
    # Filter dataset based on task requirements
    if config["task"] in [
        "AbsPositionRegressionxPDF",
        "AbsPositionRegressionXRD",
        "AbsPositionRegressionSAXS",
    ]:
        # Calculate maximum atoms the model can handle
        max_atoms = config["Model_config"]["out_channels"] // 3
        
        # Validate that all structures have fewer atoms than max_atoms
        print(f"Validating dataset (max atoms: {max_atoms})...")
        validate_dataset_atom_count(dataset.train_set, max_atoms, "train")
        validate_dataset_atom_count(dataset.validation_set, max_atoms, "validation")
        validate_dataset_atom_count(dataset.test_set, max_atoms, "test")
        print("Dataset validation successful - all structures have appropriate atom counts.")
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=config["Train_config"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset.validation_set,
        batch_size=config["Train_config"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset.test_set,
        batch_size=config["Train_config"]["batch_size"],
        shuffle=False,
    )
    
    print(f"Dataset: {config['dataset']}")
    print(f"Task: {config['task']}")
    print(f"Model: {config['model']}")
    print(f"Training samples: {len(dataset.train_set)}")
    print(f"Validation samples: {len(dataset.validation_set)}")
    print(f"Test samples: {len(dataset.test_set)}")
    
    # Create results dataframe
    results_df = pd.DataFrame(
        columns=[
            "Model",
            "Dataset",
            "Task",
            "Seed",
            "Train samples",
            "Val samples",
            "Test samples",
            "Train time",
            "Trainable parameters",
            "Train loss",
            "Metric name",
            "Val metric",
            "Test metric",
        ]
    )
    
    # Define model configurations
    model_configurations = {
        # "GCN": {
        #     "class": GCN,
        #     "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
        #     "skip_training": False,
        # },
        # "GraphSAGE": {
        #     "class": GraphSAGE,
        #     "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
        #     "skip_training": False,
        # },
        # "GIN": {
        #     "class": GIN,
        #     "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
        #     "skip_training": False,
        # },
        # "GAT": {
        #     "class": GAT,
        #     "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
        #     "skip_training": False,
        # },
        # "EdgeCNN": {
        #     "class": EdgeCNN,
        #     "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
        #     "skip_training": False,
        # },
        # "GraphUNet": {
        #     "class": GraphUNet,
        #     "kwargs": {"x": "None", "edge_index": "data.edge_index", "batch": "data.batch"},
        #     "skip_training": False,
        # },
        "MLP": {
            "class": BaselineMLP,
            "kwargs": {"x": "None", "batch": "data.batch"},
            "skip_training": False,
        },
    }
    
    # Import task functions from local benchmarking module
    from benchmark_tasks_utils import (
        pos_abs_from_saxs,
        pos_abs_from_xrd,
        pos_abs_from_xPDF,
        position_MAE
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
        # Set seed
        seed_everything(seed)
        print(f"\nRunning with seed: {seed}")
        
        # Get model configuration
        model_config = model_configurations.get(config['model'])
        if model_config is None:
            raise ValueError(f"Model {config['model']} not supported")
        
        # Create model (without Secondary)
        model_class = model_config['class']
        model_kwargs = model_config['kwargs']
        model = model_class(**config['Model_config']).to(device)
        
        # Get task configuration
        task_config = task_configurations.get(config['task'])
        if task_config is None:
            raise ValueError(f"Task {config['task']} not implemented")
        
        task_function = task_config['task_function']
        loss_function = task_config['loss_function']
        metric_function = task_config['metric_function']
        metric_name = task_config['metric_name']
        improved_function = task_config['improved_function']
        
        # Create optimizer (without Secondary parameters)
        optimizer = torch.optim.Adam(
            model.parameters(),  # Remove list() and secondary parameters
            lr=config["Train_config"]["learning_rate"],
        )
        
        # Setup logging
        save_dir = f"{config['log_dir']}/{config['dataset']}/{config['task']}/{config['model']}/seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(save_dir)
        
        # Training setup
        max_training_time = config["Train_config"]["train_time"]
        start_time = time.time()
        max_patience = config["Train_config"]["max_patience"]
        patience = 0
        best_error = None
        
        # Training loop
        for epoch in range(config['Train_config']['epochs']):
            # Check if we should skip training (for baselines)
            if model_config['skip_training']:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/best.pt",
                )
                train_loss = 0
                val_error = 0
                break
            
            # Check if we've exceeded training time
            if time.time() - start_time > max_training_time:
                print("Maximum training time reached")
                break
            
            # Check patience
            if patience >= max_patience:
                print("Maximum patience reached")
                break
            
            # Training
            model.train()
            train_loss = 0
            for data in train_loader:
                data = data.to(device)
                
                # Forward pass
                pred, truth = task_function(data, model, None, model_kwargs, device, config)  # Pass None instead of secondary
                loss = loss_function(pred, truth)
                
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
                    # print(data)
                    data = data.to(device)
                    pred, truth = task_function(data, model, None, model_kwargs, device, config) 
                    # print(pred.shape, truth.shape) # Pass None instead of secondary
                    metric = metric_function(pred, truth)
                    val_error += metric.item()
            
            val_error /= len(val_loader)
            
            # Save model if improved
            if improved_function(best_error, val_error):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/best.pt",
                )
                best_error = val_error
                patience = 0
            else:
                patience += 1
            
            # Save latest model if configured
            if config.get('save_latest_model', False):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/latest.pt",
                )
            
            # Log metrics
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar(f"{metric_name}/val", val_error, epoch)
            
            print(f"Epoch: {epoch+1}/{config['Train_config']['epochs']}, Train Loss: {train_loss:.4f}, Val {metric_name}: {val_error:.4f}")
        
        # Training complete
        stop_time = time.time()
        
        # Load best model for testing
        checkpoint = torch.load(f"{save_dir}/best.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        
        # Test evaluation
        model.eval()
        test_error = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred, truth = task_function(data, model, None, model_kwargs, device, config)  # Pass None instead of secondary
                metric = metric_function(pred, truth)
                test_error += metric.item()
        
        test_error /= len(test_loader)
        
        # Log test metrics
        writer.add_scalar(f"{metric_name}/test", test_error, epoch)
        print(f"Test {metric_name}: {test_error:.4f}")
        
        # Close writer
        writer.close()
        
        # Add results to dataframe
        results_df.loc[seed_idx] = [
            config["model"],
            config["dataset"],
            config["task"],
            seed,
            len(dataset.train_set),
            len(dataset.validation_set),
            len(dataset.test_set),
            stop_time - start_time,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            train_loss,
            metric_name,
            val_error,
            test_error,
        ]
    
    # Save results
    results_dir = f"{config['log_dir']}/{config['dataset']}/{config['task']}/{config['model']}"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f"{results_dir}/results.csv")
    print(f"Results saved to {results_dir}/results.csv")
    
    return results_df

# Function to run benchmark from a Jupyter notebook
def run_benchmark_from_notebook(config_path, use_cpu=False):
    """
    Run the benchmark from a Jupyter notebook using a config file path.
    
    Args:
        config_path (str): Path to the configuration file.
        use_cpu (bool, optional): Force CPU usage even if GPU is available. Defaults to False.
    
    Returns:
        pd.DataFrame: Results dataframe.
    """
    class Args:
        def __init__(self, config_path, cpu):
            self.config_path = config_path
            self.cpu = cpu
    
    args = Args(config_path, use_cpu)
    return run_benchmark(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for CHILI dataset")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    run_benchmark(args)
