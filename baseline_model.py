import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """
    A simple MLP baseline model for predicting atomic positions from xPDF data.
    
    This model takes xPDF data as input and predicts atomic positions
    while maintaining the [batch_size, num_atoms, 3] shape throughout.
    """
    def __init__(self, in_channels=6000, hidden_channels=512, num_atoms=100, num_layers=3, dropout=0.1):
        """
        Initialize the BaselineMLP model.
        
        Args:
            in_channels (int): Number of input features (xPDF data points)
            hidden_channels (int): Number of hidden units in each layer
            num_atoms (int): Number of atoms to predict positions for
            num_layers (int): Number of hidden layers
            dropout (float): Dropout probability
        """
        super(BaselineMLP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_atoms = num_atoms
        
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
        
        # MLP for each atom to predict its coordinates
        self.coordinate_predictor = nn.Sequential(
            nn.Linear(hidden_channels // 4, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 3)
        )
    
    def forward(self, x, batch=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels]
            batch (torch.Tensor, optional): Batch indices (not used in this model)
                                           but included for compatibility with PyG
        
        Returns:
            torch.Tensor: Predicted atomic positions of shape [batch_size, num_atoms, 3]
        """
        batch_size = x.size(0)
        
        # Extract features from xPDF data
        features = self.feature_extractor(x)  # [batch_size, hidden_channels]
        
        # Project to features for each atom
        atom_features = self.atom_projector(features)  # [batch_size, num_atoms * (hidden_channels // 4)]
        atom_features = atom_features.view(batch_size, self.num_atoms, -1)  # [batch_size, num_atoms, hidden_channels // 4]
        
        # Predict 3D coordinates for each atom
        atom_coords = self.coordinate_predictor(atom_features)  # [batch_size, num_atoms, 3]
        
        return atom_coords

