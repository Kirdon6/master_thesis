import torch
import torch.nn as nn
from typing import Literal

class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int = 1, 
        type: Literal['MLP', 'Transformer', 'CNN'] = 'MLP',
        dropout: float = 0.1
    ):
        """
        Initialize the encoder.
        
        Args:
            input_dim (int): Input dimension (sequence length)
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            num_layers (int): Number of layers
            type (str): Type of encoder ('MLP', 'Transformer', or 'CNN')
            dropout (float): Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
        """
        super(Encoder, self).__init__()
        
        # Parameter validation
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(dropout, float) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must be a float between 0 and 1, got {dropout}")
        if type not in ['MLP', 'Transformer', 'CNN']:
            raise ValueError(f"type must be one of ['MLP', 'Transformer', 'CNN'], got {type}")
            
        # Transformer-specific validation
        if type == 'Transformer':
            if input_dim % 8 != 0:  # For multi-head attention
                raise ValueError(f"For Transformer, input_dim must be divisible by 8 (num_heads), got {input_dim}")
        
        if type == 'MLP':
            # For MLP, we directly process the 1D input
            layers = []
            # First layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            # Hidden layers
            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim if i < num_layers - 2 else output_dim),
                    nn.BatchNorm1d(hidden_dim if i < num_layers - 2 else output_dim),
                    nn.ReLU() if i < num_layers - 2 else nn.Identity(),
                    nn.Dropout(dropout) if i < num_layers - 2 else nn.Identity()
                ])
            self.encoder = nn.Sequential(*layers)
            
        elif type == 'Transformer':
            # For Transformer, we need to reshape the input
            layers = []
            # Layer normalization for input
            layers.append(nn.LayerNorm(input_dim))
            
            for i in range(num_layers):
                layers.extend([
                    nn.TransformerEncoderLayer(
                        d_model=input_dim,
                        nhead=8,
                        dim_feedforward=hidden_dim,
                        dropout=dropout,
                        batch_first=True
                    ),
                    nn.LayerNorm(input_dim)
                ])
                if i < num_layers - 1:
                    layers.extend([
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
            
            # Final linear layer to output dimension
            layers.append(nn.Linear(input_dim, output_dim))
            self.encoder = nn.Sequential(*layers)
            
        elif type == 'CNN':
            # For CNN, we need to add a channel dimension
            layers = []
            # First conv layer
            layers.extend([
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            # Hidden conv layers
            for i in range(num_layers - 1):
                out_channels = hidden_dim if i < num_layers - 2 else output_dim
                layers.extend([
                    nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU() if i < num_layers - 2 else nn.Identity(),
                    nn.Dropout(dropout) if i < num_layers - 2 else nn.Identity()
                ])
            
            layers.extend([
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.BatchNorm1d(output_dim)
            ])
            self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
            
        Raises:
            ValueError: If input tensor has wrong shape
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be a torch.Tensor, got {type(x)}")
        
        if len(x.shape) != 2:
            raise ValueError(f"Input must have 2 dimensions [batch_size, sequence_length], got shape {x.shape}")
        
        # For CNN, we need to add a channel dimension
        if isinstance(self.encoder[0], nn.Conv1d):
            x = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
        
        # Forward pass
        z = self.encoder(x)
        return z


