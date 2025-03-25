import os
import yaml
import torch
import time
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter

from CHILI_centralAtoms import CHILI
from vector_diff import VectorDiffusion
from reporter import Reporter
from benchmark_tasks_utils import pos_abs_padded, position_MAE

def create_small_dataset(dataset, num_samples=5):
    """Create a small subset of the dataset with just a few samples."""
    # Get all indices
    all_indices = list(range(len(dataset)))
    
    # Select first num_samples indices
    selected_indices = all_indices[:num_samples]
    
    # Create a new dataset with just these samples
    small_dataset = torch.utils.data.Subset(dataset, selected_indices)
    
    return small_dataset

def benchmark_overfitting():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
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
            "visualization_period": 25,  # Visualize every 5 epochs
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
    
    # Split into train and validation sets (4 train, 1 validation)
    train_indices = list(range(4))
    val_indices = [4]
    train_dataset = torch.utils.data.Subset(small_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(small_dataset, val_indices)
    
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
    
    # Create model
    model = VectorDiffusion(**config["Model_config"]).to(device)
    print("\nModel Summary:")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Model architecture:")
    print(model)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["Train_config"]["learning_rate"],
        weight_decay=config["Train_config"]["weight_decay"]
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
    
    # Setup logging
    save_dir = f"{config['log_dir']}/seed{config['Train_config']['seeds'][0]}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    
    # Initialize reporter for visualization
    reporter = Reporter(config, device, save_dir)
    
    # Training setup
    start_time = time.time()
    best_error = None
    best_epoch = None
    patience = 0
    
    for epoch in range(config["Train_config"]["epochs"]):
        # Training
        model.train()
        train_loss = 0
        
        # Reset visualization flag for this epoch
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Extract ground truth positions
            pos_abs = pos_abs_padded(data, config, device)
            pos_abs_flat = pos_abs.view(pos_abs.size(0), -1)
            
            # Get and normalize xPDF data
            xPDF = data.y['xPDF']
            sct = xPDF[:,1,:]
            sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
            sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
            sct = (sct - sct_min) / (sct_max - sct_min)
            
            # Store current batch for visualization if needed
            if reporter is not None and reporter.should_visualize(epoch):
                sample_positions = pos_abs_flat[0].clone().unsqueeze(0)
                sample_xPDF = sct[0].clone().unsqueeze(0)
            
            # Calculate loss
            loss = model.loss(pos_abs_flat, sct)
            
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
                
                # Get and normalize xPDF data
                xPDF = data.y['xPDF']
                sct = xPDF[:,1,:]
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                
                # Get predictions
                pred = model(sct)
                truth = pos_abs_padded(data, config, device)
                truth_flat = truth.view(truth.size(0), -1)
                
                # Calculate MAE
                metric = position_MAE(pred, truth_flat)
                val_error += metric.item()
        
        val_error /= len(val_loader)
        
        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("PositionMAE/val", val_error, epoch)
        
        # Step the learning rate scheduler if it exists
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
        
        # Visualize if it's time
        if reporter is not None and reporter.should_visualize(epoch):
            reporter.visualize_diffusion(
                epoch,
                model,
                sample_positions,
                sample_xPDF,
                diffusion_steps=config["Model_config"]["T"]
            )
        
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
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        
        # Save latest model
        if config["save_latest_model"]:
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
        if patience >= config["Train_config"]["max_patience"]:
            print("Maximum patience reached")
            break
    
    # Close writer
    writer.close()
    
    # Print final results
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print(f"Best validation PositionMAE: {best_error:.4f} (achieved at epoch {best_epoch})")
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    benchmark_overfitting()
