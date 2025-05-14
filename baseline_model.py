import torch.nn as nn
import torch
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import wandb
from benchmark_tasks_utils import position_MAE, hausdorff_distance, atom_type_accuracy
from nano_evaluator import quick_batch_metric, quick_batch_metric_with_types, get_best_alignment_for_visualization

class BaselineMLP(nn.Module):
    """
    An MLP baseline model for predicting atomic positions and atom types from xPDF data.
    
    This model takes xPDF data as input and predicts both atomic positions
    and atom types while maintaining appropriate output shapes.
    """
    def __init__(self, in_channels=6000, hidden_channels=512, num_atoms=100, num_layers=3, 
                 dropout=0.1, num_atom_types=0):
        """
        Initialize the BaselineMLP model.
        
        Args:
            in_channels (int): Number of input features (xPDF data points)
            hidden_channels (int): Number of hidden units in each layer
            num_atoms (int): Number of atoms to predict positions for
            num_layers (int): Number of hidden layers
            dropout (float): Dropout probability
            num_atom_types (int): Number of atom type categories (0 if not predicting atom types)
        """
        super(BaselineMLP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_atoms = num_atoms
        self.num_atom_types = num_atom_types
        self.predict_atom_types = num_atom_types > 0
        
        # Feature extraction from xPDF
        feature_layers = [nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(num_layers - 2):
            feature_layers.extend([
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Project to features per atom
        self.atom_projector = nn.Linear(hidden_channels, num_atoms * hidden_channels // 4)
        
        # Enhanced MLP for each atom to predict its coordinates
        position_channels = hidden_channels // 2  # Using wider channels for positions
        
        self.position_projector = nn.Linear(hidden_channels // 4, position_channels)
        
        # First block with residual connection
        self.position_block1 = nn.Sequential(
            nn.Linear(position_channels, position_channels),
            nn.BatchNorm1d(position_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(position_channels, position_channels),
            nn.BatchNorm1d(position_channels)
        )
        
        # Second block with residual connection
        self.position_block2 = nn.Sequential(
            nn.Linear(position_channels, position_channels),
            nn.BatchNorm1d(position_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(position_channels, position_channels),
            nn.BatchNorm1d(position_channels)
        )
        
        # Position refinement block
        self.position_block3 = nn.Sequential(
            nn.Linear(position_channels, position_channels // 2),
            nn.BatchNorm1d(position_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.5)  # Less dropout in final layers
        )
        
        # Final position prediction layer
        self.position_predictor = nn.Linear(position_channels // 2, 3)
        
        # Activation for residual connections
        self.act = nn.LeakyReLU(0.2)
        
        # Enhanced MLP for each atom to predict its type (if requested)
        if self.predict_atom_types:
            # Deeper and more sophisticated atom type predictor
            atom_type_channels = hidden_channels // 2  # Using wider channels
            
            self.atom_type_projector = nn.Linear(hidden_channels // 4, atom_type_channels)
            
            # First block with residual connection
            self.atom_type_block1 = nn.Sequential(
                nn.Linear(atom_type_channels, atom_type_channels),
                nn.BatchNorm1d(atom_type_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(atom_type_channels, atom_type_channels),
                nn.BatchNorm1d(atom_type_channels)
            )
            
            # Second block with residual connection
            self.atom_type_block2 = nn.Sequential(
                nn.Linear(atom_type_channels, atom_type_channels),
                nn.BatchNorm1d(atom_type_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(atom_type_channels, atom_type_channels),
                nn.BatchNorm1d(atom_type_channels)
            )
            
            # Final classification layer
            self.atom_type_classifier = nn.Linear(atom_type_channels, num_atom_types)
    
    def forward(self, x, batch=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels]
            batch (torch.Tensor, optional): Batch indices (not used in this model)
                                           but included for compatibility with PyG
        
        Returns:
            dict: Dictionary containing:
                 'coords': Predicted atomic positions of shape [batch_size, num_atoms, 3]
                 'atom_types': Predicted atom types of shape [batch_size, num_atoms, num_atom_types]
                               (only if num_atom_types > 0)
        """
        batch_size = x.shape[0]
        
        # Extract features from xPDF data
        features = self.feature_extractor(x)  # [batch_size, hidden_channels]
        
        # Project to features for each atom
        atom_features = self.atom_projector(features)  # [batch_size, num_atoms * (hidden_channels // 4)]
        atom_features = atom_features.view(batch_size, self.num_atoms, -1)  # [batch_size, num_atoms, hidden_channels // 4]
        
        # Predict 3D coordinates for each atom using the enhanced position network
        # Reshape for batch norm layers (which expect [N, C] format)
        batch_size, num_atoms, feat_dim = atom_features.shape
        atom_features_flat = atom_features.reshape(-1, feat_dim)  # [batch_size*num_atoms, feat_dim]
        
        # Project to higher dimensional space for positions
        position_features = self.position_projector(atom_features_flat)  # [batch_size*num_atoms, position_channels]
        
        # First residual block for positions
        identity = position_features
        out = self.position_block1(position_features)
        out = self.act(out + identity)  # Add residual connection
        
        # Second residual block for positions
        identity = out
        out = self.position_block2(out)
        out = self.act(out + identity)  # Add residual connection
        
        # Final position refinement
        out = self.position_block3(out)
        atom_coords_flat = self.position_predictor(out)  # [batch_size*num_atoms, 3]
        
        # Reshape back to [batch_size, num_atoms, 3]
        atom_coords = atom_coords_flat.view(batch_size, num_atoms, 3)
        
        # Create result dictionary
        result = {'coords': atom_coords}
        
        # If predicting atom types, add atom type prediction using the enhanced network
        if self.predict_atom_types:
            # Use the same atom_features_flat we already computed
            
            # Project to higher dimensional space for atom types
            atom_type_features = self.atom_type_projector(atom_features_flat)  # [batch_size*num_atoms, atom_type_channels]
            
            # First residual block
            identity = atom_type_features
            out = self.atom_type_block1(atom_type_features)
            out = self.act(out + identity)  # Add residual connection
            
            # Second residual block
            identity = out
            out = self.atom_type_block2(out)
            out = self.act(out + identity)  # Add residual connection
            
            # Final classification
            atom_type_logits = self.atom_type_classifier(out)  # [batch_size*num_atoms, num_atom_types]
            
            # Reshape back to [batch_size, num_atoms, num_atom_types]
            atom_type_logits = atom_type_logits.view(batch_size, num_atoms, self.num_atom_types)
            
            result['atom_types'] = atom_type_logits
        
        return result
    


    
    
def train(model, optimizer, scheduler, train_data, val_data=None, test_data=None, epochs=100, batch_size=64, device='cuda', per_epoch_callback=None, use_wandb=False, wandb_run=None):
    # Create dataloader if dataset is provided
    if not isinstance(train_data, torch.utils.data.DataLoader):
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        dataloader = train_data
        
    # Create val_dataloader if a validation dataset is provided
    val_dataloader = None
    if val_data is not None:
        if not isinstance(val_data, torch.utils.data.DataLoader):
            val_dataloader = torch.utils.data.DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            val_dataloader = val_data
    
    total_steps = len(dataloader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    # Metrics to track
    train_losses = []
    val_maes = []
    val_hausdorffs = []
    val_atom_type_accuracies = []
    val_optimized_maes = []
    val_match_accuracies = []
    val_optimized_typed_maes = []
    # Setup loss functions
    pos_criterion = torch.nn.SmoothL1Loss()
    
    # Compute class weights for atom types if possible
    all_atom_types = []
    for batch in dataloader:
        positions, atom_types, conditioning = batch
        if atom_types is not None:
            all_atom_types.append(atom_types.flatten())
    
    # Initialize default atom type criterion
    atom_type_criterion = torch.nn.CrossEntropyLoss()
    
    # If we have atom types, calculate class weights for a weighted criterion
    if all_atom_types:
        all_atom_types_tensor = torch.cat(all_atom_types, dim=0)
        # Convert to long before using bincount
        all_atom_types_tensor = all_atom_types_tensor.long()
        class_counts = torch.bincount(all_atom_types_tensor)
        
        # Get the model's number of atom types
        num_atom_types = model.num_atom_types if hasattr(model, 'num_atom_types') else len(class_counts)
        
        # Ensure weights vector has correct length for all possible classes
        if len(class_counts) < num_atom_types:
            padding = torch.zeros(num_atom_types - len(class_counts), 
                                 device=class_counts.device, 
                                 dtype=class_counts.dtype)
            class_counts = torch.cat([class_counts, padding])
        
        # Handle zeros in counts to avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)
        
        class_weights = 1.0 / class_counts.float()
        # Normalize weights so they sum to number of classes
        class_weights = class_weights * (num_atom_types / class_weights.sum())
        
        # Create weighted loss
        atom_type_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        # print(f"Using weighted CrossEntropyLoss with weights of shape {class_weights.shape} for {num_atom_types} atom types")
    
    # Log model architecture to wandb if enabled
    if use_wandb and wandb_run is not None:
        # Log model architecture
        model_params = sum(p.numel() for p in model.parameters())
        wandb.run.summary["model/parameters"] = model_params
        
        # Log hyperparameters
        wandb.config.update({
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model_type": model.__class__.__name__,
            "device": str(device),
            "dataset_size": len(dataloader.dataset) if hasattr(dataloader, 'dataset') else "N/A",
        })


    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_losses = []
        epoch_pos_losses = []
        epoch_atom_type_losses = []
        
        for i, batch in enumerate(dataloader):
            # Get data - handle both formats
            positions, atom_types, conditioning = batch
            
            # Move to device
            positions = positions.to(device)
            conditioning = conditioning.to(device)
            if atom_types is not None:
                atom_types = atom_types.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(conditioning)
            
            # Position loss
            pos_loss = pos_criterion(predictions['coords'], positions)
            
            # Total loss starts with position loss
            total_loss = pos_loss
            
            # Add atom type loss if available
            if 'atom_types' in predictions and atom_types is not None:
                atom_logits = predictions['atom_types']  # [batch_size, num_atoms, num_atom_types]
                
                # Ensure atom_types has the right shape for CrossEntropyLoss
                if atom_types.ndim == 1 and atom_logits.ndim == 3:
                    # atom_types is [batch_size*num_atoms] but we need [batch_size, num_atoms]
                    if len(atom_types) == atom_logits.shape[0] * atom_logits.shape[1]:
                        atom_types = atom_types.reshape(atom_logits.shape[0], atom_logits.shape[1])
                    elif len(atom_types) == atom_logits.shape[0]:
                        # One type per batch, expand to match atoms
                        atom_types = atom_types.unsqueeze(1).expand(-1, atom_logits.shape[1])
                
                # Reshape for CrossEntropyLoss - flatten both tensors
                B, N, C = atom_logits.shape
                atom_logits_flat = atom_logits.reshape(B*N, C)
                
                # Make atom_types match the flattened logits
                if atom_types.ndim == 2 and atom_types.shape[0] == B and atom_types.shape[1] == N:
                    atom_types_flat = atom_types.reshape(B*N)
                else:
                    # Handle mismatched shapes
                    if atom_types.ndim == 1 and len(atom_types) == B:
                        # Repeat for each atom if we only have one type per batch
                        atom_types_flat = atom_types.repeat_interleave(N)
                    else:
                        # Create a default tensor
                        atom_types_flat = torch.zeros(B*N, dtype=torch.long, device=device)
                
                # Convert to long for classification
                atom_types_flat = atom_types_flat.long()
                
                atom_type_loss = atom_type_criterion(atom_logits_flat, atom_types_flat)
                
                # Combine losses with weighting
                atom_type_weight = 1.0  # Adjust as needed
                total_loss = pos_loss + atom_type_weight * atom_type_loss
                
                # Track atom type loss
                epoch_atom_type_losses.append(atom_type_loss.item())
                
                # Log separate losses
                # print(f"Position Loss: {pos_loss.item():.4f}, Atom Type Loss: {atom_type_loss.item():.4f}")
            
            # Track position loss
            epoch_pos_losses.append(pos_loss.item())
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            # Update scheduler
            scheduler.step()
            
            epoch_losses.append(total_loss.item())

            progress_bar.set_postfix(
                total_loss=f"{total_loss.item():8.4f}",
                pos_loss=f"{pos_loss.item():8.4f}",
                atom_loss=f"{atom_type_loss.item() if 'atom_types' in predictions and atom_types is not None else 0.0:8.4f}",
                epoch=f"{epoch+1}/{epochs}",
                lr=f"{scheduler.get_last_lr()[0]:.2E}"
            )

            progress_bar.update()
        
        # Calculate average training loss
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Log training metrics to wandb
        if use_wandb and wandb_run is not None:
            metrics_dict = {
                "train/loss": avg_train_loss,
                "train/epoch": epoch,
                "train/learning_rate": scheduler.get_last_lr()[0],
            }
            
            # Add component losses if available
            if epoch_pos_losses:
                avg_pos_loss = sum(epoch_pos_losses) / len(epoch_pos_losses)
                metrics_dict["train/position_loss"] = avg_pos_loss
                
            if epoch_atom_type_losses:
                avg_atom_loss = sum(epoch_atom_type_losses) / len(epoch_atom_type_losses)
                metrics_dict["train/atom_type_loss"] = avg_atom_loss
                
            wandb.log(metrics_dict, step=epoch)

        if val_dataloader is not None:
            # Validation phase
            val_metrics = validate(model, val_dataloader, device)
            val_maes.append(val_metrics['mae'])
            val_hausdorffs.append(val_metrics['hausdorff'])
            val_atom_type_accuracies.append(val_metrics['atom_type_accuracy'])
            val_optimized_maes.append(val_metrics['optimized_mae'])
            val_optimized_typed_maes.append(val_metrics['optimized_typed_mae'])


            progress_bar.set_postfix(
                loss=f"⠀{avg_train_loss:.6f}", 
                val_mae=f"{val_metrics['mae']:.6f}", 
                val_hausdorff=f"{val_metrics['hausdorff']:.6f}",
                atom_acc=f"{val_metrics['atom_type_accuracy']:.2f}%",
                epoch=f"{epoch+1}/{epochs}", 
                lr=f"{scheduler.get_last_lr()[0]:.2E}"
            )
            print(f"""\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, 
                Val MAE: {val_metrics['mae']:.6f}, Val Optimized MAE: {val_metrics['optimized_mae']:.6f}, 
                Val Optimized Typed MAE: {val_metrics['optimized_typed_mae']:.6f}, 
                Val Hausdorff: {val_metrics['hausdorff']:.6f},
                Atom Type Acc: {val_metrics['atom_type_accuracy']:.2f}%""")
            
            # Log validation metrics to wandb
            if use_wandb and wandb_run is not None:
                wandb.log({
                    "val/mae": val_metrics['mae'],
                    "val/hausdorff": val_metrics['hausdorff'],
                    "val/atom_type_accuracy": val_metrics['atom_type_accuracy'],
                    "val/optimized_mae": val_metrics['optimized_mae'],
                    "val/optimized_typed_mae": val_metrics['optimized_typed_mae'],
                    "val/epoch": epoch,
                }, step=epoch)
        
        if per_epoch_callback:
            callback_result = per_epoch_callback(model, epoch)
            
            # If callback produces visualizations, log them to wandb
            if use_wandb and wandb_run is not None and callback_result:
                # If the callback returns a dict with paths to saved files, log them
                if isinstance(callback_result, dict) and 'image_paths' in callback_result:
                    for img_name, img_path in callback_result['image_paths'].items():
                        wandb.log({f"visualizations/{img_name}": wandb.Image(img_path)}, step=epoch)
        
    return {
        'train_losses': train_losses,
        'val_maes': val_maes,
        'val_optimized_maes': val_optimized_maes,
        'val_hausdorffs': val_hausdorffs,
        'val_atom_type_accuracies': val_atom_type_accuracies,
        'val_optimized_typed_maes': val_optimized_typed_maes,
        'val_match_accuracies': val_match_accuracies
    }

def validate(model, val_dataloader, device):


    model.eval()
    all_preds = []
    all_truths = []
    all_atom_type_preds = []
    all_atom_type_truths = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Get data - handle both formats
            positions, atom_types, conditioning = batch

            # Move to device
            positions = positions.to(device)
            conditioning = conditioning.to(device)
            if atom_types is not None:
                atom_types = atom_types.to(device)
                
            # Forward pass
            predictions = model(conditioning)
            
            all_preds.append(predictions['coords'])
            all_truths.append(positions)
            
            # Collect atom type predictions if available
            if 'atom_types' in predictions and atom_types is not None:
                all_atom_type_preds.append(predictions['atom_types'])
                all_atom_type_truths.append(atom_types)
    
    # Calculate MAE for validation
    all_preds = torch.cat(all_preds, dim=0)
    all_truths = torch.cat(all_truths, dim=0)
    all_atom_type_preds = torch.cat(all_atom_type_preds, dim=0)
    all_atom_type_truths = torch.cat(all_atom_type_truths, dim=0)
    
    val_mae = position_MAE(all_preds, all_truths)
    val_optimized_mae = quick_batch_metric(all_preds, all_truths)
    typed_metrics = quick_batch_metric_with_types(all_preds, all_truths, all_atom_type_preds, all_atom_type_truths, input_format='atoms')
    optimized_typed_mae = typed_metrics['mean_distance']

    val_hausdorff = hausdorff_distance(all_preds, all_truths)
    val_accuracy = atom_type_accuracy(all_atom_type_preds, all_atom_type_truths, model_type='mlp')

    
    
    return {
            'mae': val_mae.item(),
            'optimized_mae': val_optimized_mae,
            'hausdorff': val_hausdorff,
            'atom_type_accuracy': val_accuracy,
            'optimized_typed_mae': optimized_typed_mae,
        }

def test(model, test_data, batch_size=256, device='cuda'):
    # Create dataloader if dataset is provided
    if not isinstance(test_data, torch.utils.data.DataLoader):
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        test_dataloader = test_data
    # Call validation function for testing
    test_metrics = validate(model, test_dataloader, device)
    print(f"""Test MAE: {test_metrics['mae']:.4f}, 
        Test Optimized MAE: {test_metrics['optimized_mae']:.4f}, 
        Test Optimized Typed MAE: {test_metrics['optimized_typed_mae']:.4f}, 
        Test Hausdorff: {test_metrics['hausdorff']:.4f}, 
        Test Atom Type Accuracy: {test_metrics['atom_type_accuracy']:.2f}%""")
    
    return test_metrics


def save_metrics_to_csv(metrics, filepath, model_params=None):
    import pandas as pd
    import os

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Create dictionary with final metrics
    data = {}

    # Add only final values
    if 'train_losses' in metrics and len(metrics['train_losses']) > 0:
        data['final_train_loss'] = metrics['train_losses'][-1]
    
    if 'val_maes' in metrics and len(metrics['val_maes']) > 0:
        data['final_val_mae'] = metrics['val_maes'][-1]
    
    if 'val_hausdorffs' in metrics and len(metrics['val_hausdorffs']) > 0:
        data['final_val_hausdorff'] = metrics['val_hausdorffs'][-1]

    if 'val_optimized_maes' in metrics and len(metrics['val_optimized_maes']) > 0:
        data['final_val_optimized_mae'] = metrics['val_optimized_maes'][-1]
    
    # Add atom type accuracy metrics
    if 'val_atom_type_accuracies' in metrics and len(metrics['val_atom_type_accuracies']) > 0:
        data['final_val_atom_type_accuracy'] = metrics['val_atom_type_accuracies'][-1]
    
    if 'val_optimized_typed_maes' in metrics and len(metrics['val_optimized_typed_maes']) > 0:
        data['final_val_optimized_typed_mae'] = metrics['val_optimized_typed_maes'][-1]
    
    if 'test_mae' in metrics:
        data['test_mae'] = metrics['test_mae']
    
    if 'test_hausdorff' in metrics:
        data['test_hausdorff'] = metrics['test_hausdorff']

    if 'test_optimized_mae' in metrics:
        data['test_optimized_mae'] = metrics['test_optimized_mae']
    
    # Add test atom type accuracy if available
    if 'test_atom_type_accuracy' in metrics:
        data['test_atom_type_accuracy'] = metrics['test_atom_type_accuracy']
    
    if 'test_optimized_typed_mae' in metrics:
        data['test_optimized_typed_mae'] = metrics['test_optimized_typed_mae']
    
    
    # Add model parameters if provided
    if model_params:
        data.update(model_params)
    
    # Convert to DataFrame (single row)
    df = pd.DataFrame([data])
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Final metrics saved to {filepath}")


class MLPDataset(Dataset):
    """
    Dataset for MLP model that handles both xPDF and XRD data for structure prediction.
    It processes the input data and prepares normalized features and targets.
    """
    def __init__(self, data_loader, model_type='pos_abs', cond_type='xpdf', atom_mapping=None):
        """
        Initialize the dataset from a PyG dataloader.
        
        Args:
            data_loader: PyG dataloader containing batched data
            model_type: Position prediction type ('pos_abs' or 'pos_frac')
            cond_type: Input type ('xpdf' or 'xrd')
            atom_mapping: Dictionary mapping between atom numbers and indices
        """
        self.cond_type = cond_type.lower()
        self.model_type = model_type.lower()
        self.atom_mapping = atom_mapping
        self.conditioning = []
        self.positions = []
        self.atom_types = []
        
        # Extract all data from the dataloader and store it
        for batch in data_loader:
            # Extract spectroscopy data and positions based on task
            if self.cond_type == 'xpdf':
                xpdf = batch.y['xPDF']
                sct = xpdf[:,1,:]
                # Normalize xPDF data
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                
                if 'abs' in self.model_type:
                    positions = batch.pos_abs.reshape(-1, 100, 3)
                else:  # frac
                    positions = batch.pos_frac.reshape(-1, 100, 3)
            else:  # 'xrd' in self.task
                xrd = batch.y['xrd']
                sct = xrd[:,1,:]
                # Normalize XRD data
                sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
                sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
                sct = (sct - sct_min) / (sct_max - sct_min)
                
                if 'abs' in self.model_type:
                    positions = batch.pos_abs.reshape(-1, 100, 3)
                else:  # frac
                    positions = batch.pos_frac.reshape(-1, 100, 3)
            

            if self.atom_mapping and hasattr(batch, 'x'):
                atom_numbers = batch.x[:, 0].cpu()
                # Create tensor to hold atom types
                atom_num_to_idx = self.atom_mapping['atom_num_to_idx']
                atom_indices = torch.zeros_like(atom_numbers)
                
                # Convert each atom number to its corresponding index
                for i, atom_num in enumerate(atom_numbers):
                    atom_num_int = int(atom_num.item())
                    # Default to 0 if atom type not in mapping
                    atom_indices[i] = int(atom_num_to_idx.get(str(atom_num_int), 0))
            # Store the processed data
            self.conditioning.append(sct)
            self.positions.append(positions)

            atom_indices_reshaped = atom_indices.reshape(-1, 100)
            self.atom_types.append(atom_indices_reshaped)
    
        # Concatenate all batches
        self.conditioning = torch.cat(self.conditioning, dim=0)
        self.positions = torch.cat(self.positions, dim=0)
        if self.atom_types:
            self.atom_types = torch.cat(self.atom_types, dim=0)
            # Ensure shapes are compatible
            if self.positions.shape[0] != self.atom_types.shape[0] or self.positions.shape[1] != self.atom_types.shape[1]:
                raise ValueError(f"Shape mismatch: positions shape {self.positions.shape}, atom_types shape {self.atom_types.shape}")
    
    def __len__(self):
        return len(self.conditioning)
    
    def __getitem__(self, idx):
        if len(self.atom_types) > 0:
            return (self.positions[idx], self.atom_types[idx], self.conditioning[idx])
        else:
            return (self.positions[idx], self.conditioning[idx])

def train_mlp_model(train_loader, val_loader, test_loader, sample_dir=None, cond_type='xPDF', in_channels=6000, 
                    hidden_dim=512, num_layers=3, dropout=0.1, learning_rate=0.001, epochs=200, batch_size=64, 
                    weight_decay=0.0001, lr_scheduler=True, lr_step_size=30, lr_gamma=0.5, 
                   atom_mapping_path=None, model_type='pos_frac', use_wandb=False, wandb_run=None):
    """
    Train and evaluate the MLP baseline model.
    
    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    sample_dir : str
        Directory to save visualizations
    cond_type : str
        Conditioning type (xPDF or XRD)
    use_wandb : bool
        Whether to use Weights & Biases for logging
    wandb_run : wandb.run
        Existing wandb run to use for logging
    **kwargs : dict
        Additional model and training parameters
        
    Returns
    -------
    model : BaselineMLP
        Trained MLP model
    metrics : dict
        Dictionary of training and evaluation metrics
    """
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    atom_suffix = "_with_atoms" if atom_mapping_path else ""
    model_params = f"{model_type}_layers{num_layers}_hidden{hidden_dim}_lr{learning_rate}_epochs{epochs}_batch{batch_size}_cond{in_channels}{atom_suffix}_{timestamp}"
    samples_dir = os.path.join(sample_dir, "training_samples", model_params)
    
    if not os.path.exists(os.path.join(sample_dir, "training_samples")):
        os.makedirs(os.path.join(sample_dir, "training_samples"))
    
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Load atom type mapping if provided
    atom_mapping = None
    if atom_mapping_path:
        try:
            import json
            with open(atom_mapping_path, 'r') as f:
                atom_mapping = json.load(f)
            num_atom_types = atom_mapping['num_categories']
            print(f"Loaded atom mapping with {num_atom_types} categories for visualization")
        except Exception as e:
            print(f"Warning: Could not load atom mapping for visualization: {e}")
            num_atom_types = 0
    
    # Create the dataset wrappers
    train_dataset = MLPDataset(train_loader, model_type=model_type, cond_type=cond_type, atom_mapping=atom_mapping)
    val_dataset = MLPDataset(val_loader, model_type=model_type, cond_type=cond_type, atom_mapping=atom_mapping)
    test_dataset = MLPDataset(test_loader, model_type=model_type, cond_type=cond_type, atom_mapping=atom_mapping)
    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # Create the MLP model with atom type prediction
    model = BaselineMLP(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        num_atoms=100,
        num_layers=num_layers,
        dropout=dropout,
        num_atom_types=num_atom_types
    ).to(device)
    
    logging.info(f"Created BaselineMLP with {num_layers} layers on {device}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)  # No-op scheduler
    

    # Save reference to first few images for sampling comparison
    # Get the first batch from the train dataset for visualization
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,  # Only need a small batch for visualization
        shuffle=True
    )

    # Initialize storage for ground truth data
    ground_truth_positions = []
    ground_truth_atom_types = []
    ground_truth_conditioning = []
    
    # Get reference data for visualization
    for batch in train_dataloader:
        positions, atom_types, conditioning = batch
        
        # Store each structure in the batch individually
        for i in range(positions.shape[0]):
            ground_truth_positions.append(positions[i:i+1].to(device))  # Add batch dimension
            ground_truth_conditioning.append(conditioning[i:i+1].to(device))  # Add batch dimension
            if atom_types is not None:
                ground_truth_atom_types.append(atom_types[i:i+1].to(device))  # Add batch dimension
        
        break  # Only need one batch

    
    # Function to visualize ground truth vs predictions
    def visualize_predictions(model, epoch):
        """Create a visualization of ground truth vs predicted structures with atom types"""
        # Skip most visualizations to save time and space if using wandb
        if use_wandb and wandb_run is not None and epoch % 10 != 0 and epoch != epochs - 1:
            return
            
        model.eval()
        
        with torch.no_grad():
            # Store all 3D sample points for plotting
            all_gt_points = []
            all_sample_points = []
            all_gt_atom_types = []
            all_sample_atom_types = []
            all_aligned_samples = []

            for gt_idx in range(len(ground_truth_positions)):
                gt_points = ground_truth_positions[gt_idx]
                cond = ground_truth_conditioning[gt_idx]
                gt_atom_types = ground_truth_atom_types[gt_idx]

                # Get model predictions
                predictions = model(cond)

                sample_points = predictions['coords']
                sample_atom_types = predictions.get('atom_types', None)

                gt_points = gt_points.cpu()
                sample_points = sample_points.cpu()
                gt_atom_types = gt_atom_types.cpu()
                
                if sample_atom_types is not None:
                    sample_atom_types = sample_atom_types.cpu()

                # Get the best alignment for visualization
                # We need to properly reshape the tensors for the Kabsch alignment
                # The function expects tensors of shape [batch_size, n_atoms, 3]
                try:
                    # Reshape sample and ground truth points for alignment
                    # For model output, ensure it's [batch, atoms, 3]
                    sample_for_align = sample_points.clone()
                    if sample_for_align.ndim == 2:  # [atoms, 3]
                        sample_for_align = sample_for_align.unsqueeze(0)  # Add batch dimension [1, atoms, 3]
                    
                    # For ground truth, ensure it's [batch, atoms, 3]
                    gt_for_align = gt_points.clone()
                    if gt_for_align.ndim == 2:  # [atoms, 3]
                        gt_for_align = gt_for_align.unsqueeze(0)  # Add batch dimension [1, atoms, 3]
                    
                    # Perform alignment without atom types to avoid type errors
                    aligned_structures = get_best_alignment_for_visualization(sample_for_align, gt_for_align)
                    all_aligned_samples.append(aligned_structures)
                except Exception as e:
                    print(f"Error during alignment: {e}")
                    # Create a dummy alignment result structure to avoid further errors
                    all_aligned_samples.append({
                        'aligned_pred_coords': sample_points.unsqueeze(0),  # Just use original coords
                        'true_coords': gt_points.unsqueeze(0),
                        'pred_types': sample_atom_types,
                        'true_types': gt_atom_types
                    })

                # Store 3D sample points for plotting
                all_gt_points.append(gt_points)
                all_sample_points.append(sample_points)
                
                # Process ground truth atom types
                if gt_atom_types is not None:
                    # Make sure atom_types has the right format for plotting
                    if gt_atom_types.ndim > 1:
                        # For multi-dimensional tensors, get indices of max values to use as colors
                        if gt_atom_types.shape[-1] > 1 and len(gt_atom_types.shape) > 2:
                            gt_atom_types = torch.argmax(gt_atom_types, dim=-1)
                    all_gt_atom_types.append(gt_atom_types)
                
                # Process sample atom types
                if sample_atom_types is not None:
                    # For predictions, get class indices from logits
                    if sample_atom_types.ndim > 2:
                        # If it's [batch, atoms, num_classes], get the predicted class indices
                        sample_atom_types = torch.argmax(sample_atom_types, dim=-1)
                    elif sample_atom_types.ndim == 2 and sample_atom_types.shape[-1] > 1:
                        # If it's [batch, num_classes] or [atoms, num_classes], get the predicted class
                        if sample_atom_types.shape[0] == sample_points.shape[0]:
                            # One logit vector per structure/batch
                            sample_atom_classes = torch.argmax(sample_atom_types, dim=-1)
                            # We need to repeat this for each atom in the structure
                            if sample_points.ndim == 3:
                                # Expand to match shape [batch, atoms]
                                sample_atom_classes = sample_atom_classes.unsqueeze(1).expand(-1, sample_points.shape[1])
                            sample_atom_types = sample_atom_classes
                    
                    all_sample_atom_types.append(sample_atom_types)

                
            
            # Create 3x2 grid for visualization (2 rows, 3 columns)
            fig = plt.figure(figsize=(24, 12), facecolor='white')
            
            # Function to style 3D axes
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
            
            # Create colormap for atom types if needed
            if all_gt_atom_types and num_atom_types > 0:
                # Create appropriate colormap based on number of atom types
                if num_atom_types <= 10:
                    # For 10 or fewer categories, tab10 is excellent
                    cmap = plt.cm.get_cmap('tab10', num_atom_types)
                elif num_atom_types <= 20:
                    # For up to 20, we can use tab20
                    cmap = plt.cm.get_cmap('tab20', num_atom_types)
                else:
                    # For more categories, create a custom colormap with enough distinct colors
                    # Create evenly spaced hues
                    hues = np.linspace(0, 1, num_atom_types, endpoint=False)
                    # Create colors with varying hue, full saturation and value
                    hsv_colors = [(h, 0.8, 0.9) for h in hues]
                    # Convert HSV to RGB
                    rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
                    # Create a ListedColormap
                    cmap = mcolors.ListedColormap(rgb_colors)
                
                # Use BoundaryNorm to get discrete color levels
                bounds = np.arange(0, num_atom_types+1)
                norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
                # Add a special axis for the colorbar on the left
                cbar_ax = fig.add_axes([0.05, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(sm, cax=cbar_ax)
                cbar.set_label('Atom Type', size=12)
                
                # Add atom type labels if available
                if atom_mapping:
                    idx_to_atom_num = atom_mapping.get('idx_to_atom_num', {})
                    # Create labels for each category
                    labels = []
                    for i in range(num_atom_types):
                        atom_num = idx_to_atom_num.get(str(i), "?")
                        labels.append(f"Z={atom_num}")
                    
                    # Set tick positions and labels for a discrete colorbar
                    ticks = np.arange(num_atom_types) + 0.5
                    
                    # If there are too many atom types, show only a subset
                    if num_atom_types > 20:
                        stride = max(1, num_atom_types // 15)
                        subset_indices = range(0, num_atom_types, stride)
                        subset_ticks = [ticks[i] for i in subset_indices]
                        subset_labels = [labels[i] for i in subset_indices]
                        cbar.set_ticks(subset_ticks)
                        cbar.set_ticklabels(subset_labels)
                    else:
                        cbar.set_ticks(ticks)
                        cbar.set_ticklabels(labels)
            
            # Plot each row (one structure per row, with three different views: GT, prediction, aligned)
            for i in range(min(2, len(all_gt_points))):
                # Ground truth (1st column)
                ax_gt = fig.add_subplot(2, 3, i*3+1, projection='3d')
                gt_points = all_gt_points[i]
                
                # If atom types are available, use them for coloring
                if i < len(all_gt_atom_types) and num_atom_types > 0:
                    gt_atom_colors = all_gt_atom_types[i].numpy()
                    
                    # Keep this essential code for flattening and reshaping
                    if gt_points.ndim == 3:  # If points is [batch, atoms, xyz]
                        # We need to flatten the points for scatter
                        num_atoms_per_structure = gt_points.shape[1]
                        gt_points_flat = gt_points.reshape(-1, 3)  # Flatten to [batch*atoms, xyz]
                        
                        # If atom colors is not the right shape, we need to fix it
                        if gt_atom_colors.ndim == 1 and len(gt_atom_colors) != len(gt_points_flat):
                            if len(gt_atom_colors) == 1:
                                # Just one color for all atoms
                                gt_atom_colors = np.full(len(gt_points_flat), gt_atom_colors[0])
                            elif len(gt_atom_colors) == gt_points.shape[0]:
                                # One color per structure, repeat for each atom
                                gt_atom_colors = np.repeat(gt_atom_colors, num_atoms_per_structure)
                            else:
                                # Try to reshape or use a default color
                                gt_atom_colors = np.zeros(len(gt_points_flat), dtype=int)
                        
                        # Final safety check - make sure the colors array is a flat 1D array
                        if gt_atom_colors.ndim != 1:
                            gt_atom_colors = gt_atom_colors.flatten()
                        
                        # And ensure it has exactly the same length as the number of points
                        if len(gt_atom_colors) != len(gt_points_flat):
                            # If too long, truncate; if too short, pad with zeros
                            if len(gt_atom_colors) > len(gt_points_flat):
                                gt_atom_colors = gt_atom_colors[:len(gt_points_flat)]
                            else:
                                # Pad with zeros
                                gt_atom_colors = np.pad(gt_atom_colors, 
                                                     (0, len(gt_points_flat) - len(gt_atom_colors)), 
                                                     mode='constant', 
                                                     constant_values=0)
                        
                        ax_gt.scatter(gt_points_flat[:, 0], gt_points_flat[:, 1], gt_points_flat[:, 2],
                                    c=gt_atom_colors, cmap=cmap, norm=norm, marker='o', s=25, alpha=0.8)
                    else:
                        # If atom colors don't match the points, use a fixed color
                        if len(gt_atom_colors) != len(gt_points):
                            ax_gt.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                                      c='blue', marker='o', s=25, alpha=0.8)
                        else:
                            ax_gt.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                                      c=gt_atom_colors, cmap=cmap, norm=norm, marker='o', s=25, alpha=0.8)
                else:
                    ax_gt.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                                c='blue', marker='o', s=25, alpha=0.8)
                
                style_3d_axes(ax_gt, f'Ground Truth Structure {i+1}')
                
                # Original Prediction (2nd column)
                ax_pred = fig.add_subplot(2, 3, i*3+2, projection='3d')
                pred_points = all_sample_points[i]
                
                # If atom type predictions are available
                if i < len(all_sample_atom_types) and num_atom_types > 0:
                    pred_atom_types = all_sample_atom_types[i].numpy()
                    
                    # Keep all the reshaping code
                    if pred_points.ndim == 3:  # If points is [batch, atoms, xyz]
                        # We need to flatten the points for scatter
                        num_atoms_per_structure = pred_points.shape[1]
                        pred_points_flat = pred_points.reshape(-1, 3)  # Flatten to [batch*atoms, xyz]
                        
                        # If atom colors is not the right shape, we need to fix it
                        if pred_atom_types.ndim > 1:
                            # The prediction has shape [batch, num_classes] or [batch, atoms, num_classes]
                            # We need to convert it to a 1D array with the same length as pred_points_flat
                            
                            if pred_atom_types.shape == (pred_points.shape[0], pred_points.shape[1]):
                                # It's already the right shape before flattening
                                pred_atom_types = pred_atom_types.reshape(-1)
                            elif pred_atom_types.shape[0] == pred_points.shape[0]:
                                # Need to expand to match atom count
                                if pred_atom_types.shape[1] != pred_points.shape[1]:
                                    # Create a default array matching the points count
                                    fixed_atom_types = np.zeros(len(pred_points_flat), dtype=int)
                                    # Use first class from each batch if there are multiple
                                    if pred_atom_types.ndim == 2:
                                        for b in range(pred_points.shape[0]):
                                            start_idx = b * num_atoms_per_structure
                                            end_idx = start_idx + num_atoms_per_structure
                                            fixed_atom_types[start_idx:end_idx] = pred_atom_types[b, 0]
                                    pred_atom_types = fixed_atom_types
                                else:
                                    # Just flatten the array if shapes match
                                    pred_atom_types = pred_atom_types.reshape(-1)
                            else:
                                # Create a default array 
                                pred_atom_types = np.zeros(len(pred_points_flat), dtype=int)
                        
                        # Ensure pred_atom_types is 1D and matches points length
                        if pred_atom_types.ndim > 1 or len(pred_atom_types) != len(pred_points_flat):
                            pred_atom_types = np.zeros(len(pred_points_flat), dtype=int)
                        
                        # Final safety check - make sure the colors array is a flat 1D array
                        if pred_atom_types.ndim != 1:
                            pred_atom_types = pred_atom_types.flatten()
                        
                        # And ensure it has exactly the same length as the number of points
                        if len(pred_atom_types) != len(pred_points_flat):
                            # If too long, truncate; if too short, pad with zeros
                            if len(pred_atom_types) > len(pred_points_flat):
                                pred_atom_types = pred_atom_types[:len(pred_points_flat)]
                            else:
                                # Pad with zeros
                                pred_atom_types = np.pad(pred_atom_types, 
                                                     (0, len(pred_points_flat) - len(pred_atom_types)), 
                                                     mode='constant', 
                                                     constant_values=0)
                        
                        ax_pred.scatter(pred_points_flat[:, 0], pred_points_flat[:, 1], pred_points_flat[:, 2],
                                    c=pred_atom_types, cmap=cmap, norm=norm, marker='o', s=25, alpha=0.8)
                    else:
                        # If atom colors don't match the points, use a fixed color
                        if len(pred_atom_types) != len(pred_points):
                            ax_pred.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                                      c='red', marker='o', s=25, alpha=0.8)
                        else:
                            ax_pred.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                                      c=pred_atom_types, cmap=cmap, norm=norm, marker='o', s=25, alpha=0.8)
                else:
                    ax_pred.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                                  c='red', marker='o', s=25, alpha=0.8)
                
                style_3d_axes(ax_pred, f'Original Prediction {i+1}')
                
                # Aligned Prediction (3rd column)
                ax_aligned = fig.add_subplot(2, 3, i*3+3, projection='3d')
                
                try:
                    aligned_structure = all_aligned_samples[i]
                    
                    # Get the aligned coordinates from the result
                    aligned_points = aligned_structure['aligned_pred_coords'][0]  # Get the first batch element
                    
                    # Ensure aligned_points is in the right format for plotting
                    if aligned_points.ndim == 3:  # [batch, n_atoms, 3]
                        aligned_points = aligned_points[0]  # Get first batch element if batched
                    
                    # IMPORTANT: Use the same atom types from the original prediction for coloring
                    # This ensures the aligned visualization matches the color scheme of the original
                    if i < len(all_sample_atom_types) and num_atom_types > 0:
                        # Use the same atom types as used for the original prediction
                        aligned_atom_types = all_sample_atom_types[i].numpy() 
                        
                        # Reshape atom types to match the aligned points
                        if aligned_atom_types.ndim == 1 and len(aligned_atom_types) == 1:
                            # Case where we have a single atom type value for all atoms
                            # Expand it to match the number of points
                            aligned_atom_types = np.full(len(aligned_points), aligned_atom_types[0])
                        elif aligned_atom_types.ndim > 1:
                            # Flatten multi-dimensional atom types
                            aligned_atom_types = aligned_atom_types.flatten()
                            
                            # If still not the right length, reshape by repeating or truncating
                            if len(aligned_atom_types) != len(aligned_points):
                                if len(aligned_atom_types) < len(aligned_points):
                                    # Repeat pattern to fill
                                    repeats = int(np.ceil(len(aligned_points) / len(aligned_atom_types)))
                                    aligned_atom_types = np.tile(aligned_atom_types, repeats)[:len(aligned_points)]
                                else:
                                    # Truncate
                                    aligned_atom_types = aligned_atom_types[:len(aligned_points)]
                        
                        # If atom colors still don't match the points, use a fixed color
                        if len(aligned_atom_types) != len(aligned_points):
                            print(f"Shape mismatch after reshaping: atom_types: {aligned_atom_types.shape}, points: {aligned_points.shape}")
                            ax_aligned.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
                                          c='green', marker='o', s=25, alpha=0.8)
                        else:
                            ax_aligned.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
                                          c=aligned_atom_types, cmap=cmap, norm=norm, marker='o', s=25, alpha=0.8)
                    else:
                        ax_aligned.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
                                      c='green', marker='o', s=25, alpha=0.8)
                except Exception as e:
                    print(f"Error visualizing aligned structure: {e}")
                    # Add a text annotation explaining the error
                    ax_aligned.text(0, 0, 0, "Alignment Error", fontsize=12)
                
                style_3d_axes(ax_aligned, f'Aligned Prediction {i+1}')
            
            # Adjust layout for better spacing
            plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15)
            
            # Save visualization
            filename = os.path.join(samples_dir, f"epoch_{epoch:03d}_structures.png")
            plt.savefig(filename, dpi=200)
            plt.close()
            
            # Log the visualization to wandb if enabled
            if use_wandb and wandb_run is not None:
                wandb.log({
                    "visualizations/3d_structures": wandb.Image(filename),
                    "epoch": epoch
                })
                
            # Return dict of image paths for potential wandb logging
            return {
                "image_paths": {
                    "3d_structures": filename
                }
            }
    
    # Training loop
    logging.info(f"Starting MLP training for {epochs} epochs")

    metrics = train(
        model,
        optimizer,
        scheduler, 
        train_dataset, 
        val_dataset, 
        test_dataset,
        epochs=epochs, 
        batch_size=batch_size, 
        device=device, 
        per_epoch_callback=visualize_predictions,
        use_wandb=use_wandb,
        wandb_run=wandb_run)
    
    
    
    # Load best model for final evaluation
    model_path = os.path.join(samples_dir, "mlp_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to '{model_path}'")

    # When testing, store the test metrics to include in final report
    test_result_metrics = {}
    if test_dataset is not None:
        model.load_state_dict(torch.load(os.path.join(samples_dir, "mlp_model.pt")))
        test_metrics = test(model, test_dataset, batch_size=batch_size, device=device)
        print(f"Final test MAE: {test_metrics['mae']:.4f}, Test Hausdorff: {test_metrics['hausdorff']:.4f}, Test Optimized MAE: {test_metrics['optimized_mae']:.4f}, Test Optimized Typed MAE: {test_metrics['optimized_typed_mae']:.4f}, Test Atom Type Accuracy: {test_metrics['atom_type_accuracy']:.2f}%")
        
        # Store test metrics for CSV reporting
        test_result_metrics = {
            'test_mae': test_metrics['mae'],
            'test_hausdorff': test_metrics['hausdorff'],
            'test_optimized_mae': test_metrics['optimized_mae'],
            'test_optimized_typed_mae': test_metrics['optimized_typed_mae'],
            'test_atom_type_accuracy': test_metrics['atom_type_accuracy']
        }
        
        # Log final test metrics to wandb
        if use_wandb and wandb_run is not None:
            wandb.log({
                "test/mae": test_metrics['mae'],
                "test/hausdorff": test_metrics['hausdorff'],
                "test/atom_type_accuracy": test_metrics['atom_type_accuracy'],
                "test/optimized_mae": test_metrics['optimized_mae'],
                "test/optimized_typed_mae": test_metrics['optimized_typed_mae'],
            })
    
    model_parameters = {
        'model_type': model_type,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'cond_type': cond_type,
        'in_channels': in_channels,
        'atom_mapping_path': atom_mapping_path,
        'device': device,
    }
    
    # Combine training metrics with test metrics for CSV
    combined_metrics = {**metrics, **test_result_metrics}
    
    metrics_path = os.path.join(samples_dir, "final_metrics.csv")
    save_metrics_to_csv(combined_metrics, metrics_path, model_parameters)
    
    # Log model architecture summary to wandb if enabled
    if use_wandb and wandb_run is not None:
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Add model summary to wandb
        wandb.run.summary.update({
            "model/total_parameters": total_params,
            "model/type": "MLP",
            "model/num_layers": num_layers,
            "model/hidden_dim": hidden_dim,
            "training/final_loss": metrics['train_losses'][-1] if metrics['train_losses'] else None,
            "results/best_val_mae": min(metrics['val_maes']) if metrics['val_maes'] else None,
            "results/final_test_mae": test_result_metrics.get('test_mae', None),
        })
        
        # Save model to wandb
        wandb.save(model_path)
    
    return model, metrics 