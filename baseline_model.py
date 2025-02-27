import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """
    A simple MLP baseline model for predicting atomic positions from xPDF data.
    
    This model takes xPDF data as input and directly predicts the flattened 
    atomic positions (x, y, z coordinates for each atom).
    """
    def __init__(self, in_channels=6000, hidden_channels=512, out_channels=300, num_layers=3, dropout=0.1):
        """
        Initialize the BaselineMLP model.
        
        Args:
            in_channels (int): Number of input features (xPDF data points)
            hidden_channels (int): Number of hidden units in each layer
            out_channels (int): Number of output features (3 * max_atoms)
            num_layers (int): Number of hidden layers
            dropout (float): Dropout probability
        """
        super(BaselineMLP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Input layer
        layers = [nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout)]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, batch=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels]
            batch (torch.Tensor, optional): Batch indices (not used in this model)
                                           but included for compatibility with PyG
        
        Returns:
            torch.Tensor: Predicted atomic positions of shape [batch_size, out_channels]
        """
        return self.model(x)

