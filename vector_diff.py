import torch
import torch.nn as nn
from encoder import Encoder

class VectorPosNet(nn.Module):
    """UNet-like neural network for predicting vector positions"""
    def __init__(self, cond_dim, hidden_dim, pos_dim):
        super(VectorPosNet, self).__init__()

        # Calculate the number of atoms from the position dimension
        self.num_atoms = pos_dim // 3
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition embedding
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initial projection from position space to feature space
        self.input_proj = nn.Linear(3, hidden_dim)
        
        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.down2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
        )
        
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.down3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
        )
        
        self.pool3 = nn.AvgPool1d(kernel_size=5, stride=5)  # From 25 to 5 atoms
        
        # Bottleneck, combining with time and condition
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim*4 + hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*4),
            nn.ReLU(),
        )
        
        # Upsampling path
        self.up3 = nn.Linear(hidden_dim*4, hidden_dim*4*5)  # Expand features for reshaping
        self.up3_conv = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
        )
        
        self.up2 = nn.Linear(hidden_dim*2, hidden_dim*2*2)  # Expand features for reshaping
        self.up2_conv = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.up1 = nn.Linear(hidden_dim, hidden_dim*2)  # Expand features for reshaping
        self.up1_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Final projection back to position space
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, pos, time, cond):
        # Ensure time is properly shaped for the embedding
        if time.dim() == 1:
            time = time.unsqueeze(1).float()
        
        # Ensure positions are shaped [batch_size, num_atoms, 3]
        batch_size = pos.shape[0]
        
        # Create embeddings
        time_emb = self.time_embedding(time)  # [batch, hidden_dim]
        cond_emb = self.cond_embedding(cond)  # [batch, hidden_dim]
        
        # Initial projection of positions
        x = self.input_proj(pos)  # [batch, 100, hidden_dim]
        
        # Add time embedding at the start
        # Expand time embedding to match atom dimension
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, pos.shape[1], -1)  # [batch, 100, hidden_dim]
        
        # Element-wise addition with position features
        x = x + time_emb_expanded  # [batch, 100, hidden_dim]
        
        # Downsampling path
        x1 = self.down1(x)  # [batch, 100, hidden_dim]
        
        # Apply pooling along atom dimension
        x1_pool = x1.transpose(1, 2)  # [batch, hidden_dim, 100]
        x1_pool = self.pool1(x1_pool)  # [batch, hidden_dim, 50]
        x1_pool = x1_pool.transpose(1, 2)  # [batch, 50, hidden_dim]
        
        x2 = self.down2(x1_pool)  # [batch, 50, hidden_dim*2]
        
        x2_pool = x2.transpose(1, 2)  # [batch, hidden_dim*2, 50]
        x2_pool = self.pool2(x2_pool)  # [batch, hidden_dim*2, 25]
        x2_pool = x2_pool.transpose(1, 2)  # [batch, 25, hidden_dim*2]
        
        x3 = self.down3(x2_pool)  # [batch, 25, hidden_dim*4]
        
        x3_pool = x3.transpose(1, 2)  # [batch, hidden_dim*4, 25]
        x3_pool = self.pool3(x3_pool)  # [batch, hidden_dim*4, 5]
        x3_pool = x3_pool.transpose(1, 2)  # [batch, 5, hidden_dim*4]
        
        # Bottleneck - combine with condition embedding only
        # Expand cond_emb to match atom dimension
        cond_emb_expanded = cond_emb.unsqueeze(1).expand(-1, x3_pool.size(1), -1)  # [batch, 5, hidden_dim]
        x_bottleneck = torch.cat([x3_pool, cond_emb_expanded], dim=2)  # [batch, 5, hidden_dim*4 + hidden_dim]
        x_bottleneck = self.bottleneck(x_bottleneck)  # [batch, 5, hidden_dim*4]
        
        # Upsampling path
        x = self.up3(x_bottleneck)  # [batch, 5, feature_dim]
        x = x.reshape(batch_size, 25, x.size(2)//5)  # [batch, 25, feature_dim]
        x = self.up3_conv(x)  # [batch, 25, reduced_dim]
        
        x = self.up2(x)  # [batch, 25, expanded_dim]
        x = x.reshape(batch_size, 50, x.size(2)//2)  # [batch, 50, reduced_dim]
        x = self.up2_conv(x)  # [batch, 50, base_dim]
        
        x = self.up1(x)  # [batch, 50, expanded_dim]
        x = x.reshape(batch_size, 100, x.size(2)//2)  # [batch, 100, base_dim]
        x = self.up1_conv(x)  # [batch, 100, base_dim]
        
        # Final projection back to position space
        x = self.output_proj(x)  # [batch, 100, 3]
        
        return x
    

class VectorDiffusion(nn.Module):
    """Diffusion model for vectors of positions"""
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super(VectorDiffusion, self).__init__()
        
        # Extract diffusion parameters from kwargs or use defaults
        self.T = kwargs.get('T', 100)
        self.beta_1 = kwargs.get('beta_1', 1e-4)
        self.beta_T = kwargs.get('beta_T', 2e-2)
        
        # Store number of atoms
        self.num_atoms = out_channels // 3
        
        # Create encoder for xPDF data - now accepts 1D input
        self.encoder = Encoder(
            input_dim=in_channels,  # This is the length of the xPDF y-values (6000)
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            num_layers=2,
            type='MLP'
        )
        
        # Create denoiser network
        self.denoiser = VectorPosNet(
            cond_dim=hidden_channels,
            hidden_dim=hidden_channels,
            pos_dim=out_channels
        )
        
        # Register diffusion parameters as buffers so they move to the right device
        self.register_buffer("betas", torch.linspace(self.beta_1, self.beta_T, self.T+1))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def forward(self, x, batch=None):
        """
        Forward pass compatible with benchmark framework
        
        Args:
            x: Input xPDF data with shape [batch_size, 6000]
            batch: Batch indices (optional)
            
        Returns:
            Predicted atom positions with shape [batch_size, num_atoms, 3]
        """
        # During inference, sample from the model
        batch_size = x.shape[0]
        
        # Define output shape for atomic positions
        shape = (batch_size, self.num_atoms, 3)
        
        # Sample from the model
        positions = self.sample(shape, x)

        return positions

    def forward_diffusion(self, x_0, t, epsilon):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: Initial positions with shape [batch_size, num_atoms, 3]
            t: Timestep
            epsilon: Random noise with same shape as x_0
            
        Returns:
            Noisy positions at timestep t
        """
        # Ensure proper broadcasting by reshaping time-dependent parameters
        batch_size = x_0.shape[0]
        alpha_bar_t = self.alpha_bars[t].view(batch_size, 1, 1)
        
        # Calculate mean and standard deviation
        mean = torch.sqrt(alpha_bar_t) * x_0
        std = torch.sqrt(1.0 - alpha_bar_t)
        
        return mean + std * epsilon
    
    def reverse_diffusion(self, x_t, t, epsilon, cond):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        
        Args:
            x_t: Positions at timestep t with shape [batch_size, num_atoms, 3]
            t: Timestep
            epsilon: Random noise with same shape as x_t
            cond: Conditioning data (xPDF) with shape [batch_size, sequence_length]
            
        Returns:
            Positions at timestep t-1
        """
        
        # Get latent embedding from encoder
        latent_emb = self.encoder(cond)
        
        # Normalize timestep for the denoiser
        t_normalized = t.float() / self.T
        
        # Predict noise using denoiser
        predicted_noise = self.denoiser(x_t, t_normalized, latent_emb)
        
        # Get batch size
        batch_size = x_t.shape[0]
        
        # Extract parameters for the current timestep and reshape for broadcasting to [batch, 1, 1]
        alpha_t = self.alphas[t].view(batch_size, 1, 1)
        beta_t = self.betas[t].view(batch_size, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(batch_size, 1, 1)
        
        # For t > 0, we also need alpha_bar_{t-1}
        alpha_bar_t_prev = torch.zeros_like(alpha_bar_t)
        mask = (t > 0)
        if torch.any(mask):
            t_prev = torch.clamp(t - 1, min=0)
            alpha_bar_t_prev[mask] = self.alpha_bars[t_prev[mask]].view(-1, 1, 1)
        
        # Calculate mean for the reverse process
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )
        
        # Calculate variance for the reverse process
        var = torch.where(
            t.view(batch_size, 1, 1) > 0, 
            beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t), 
            torch.zeros_like(beta_t)
        )
        std = torch.sqrt(var)
        
        return mean + std * epsilon
    
    def elbo_simple(self, x_0, cond):
        """
        Calculate ELBO (Evidence Lower Bound) for training
        
        Args:
            x_0: Ground truth positions with shape [batch_size, num_atoms*3]
            cond: Conditioning data (xPDF) with shape [batch_size, 6000]
            
        Returns:
            ELBO value
        """
        # Reshape x_0 to [batch_size, num_atoms, 3] if it's flattened
        batch_size = x_0.shape[0]
        if x_0.dim() == 2:
            x_0 = x_0.view(batch_size, self.num_atoms, 3)
        
        # Sample timestep
        t = torch.randint(1, self.T, (batch_size,), device=x_0.device)
        
        # Get latent embedding
        latent_emb = self.encoder(cond)
        
        # Sample noise
        epsilon = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.forward_diffusion(x_0, t, epsilon)
        
        # Normalize timestep for the denoiser
        t_normalized = t.float() / self.T
        
        # Predict noise
        predicted_noise = self.denoiser(x_t, t_normalized.unsqueeze(1), latent_emb)
        
        # Calculate loss
        return -nn.MSELoss(reduction='mean')(epsilon, predicted_noise)
    
    def loss(self, x_0, cond):
        """
        Calculate loss for training
        
        Args:
            x_0: Ground truth positions with shape [batch_size, num_atoms*3]
            cond: Conditioning data (xPDF) with shape [batch_size, 6000]
            
        Returns:
            Loss value
        """
        return -self.elbo_simple(x_0, cond)

    @torch.no_grad()
    def sample(self, shape, cond=None):
        """
        Sample atom positions from the diffusion model
        
        Args:
            shape: Shape of the output tensor [batch_size, num_atoms, 3]
            cond: Conditioning data (xPDF) with shape [batch_size, 6000]
            
        Returns:
            Sampled atom positions with shape [batch_size, num_atoms, 3]
        """
        if cond is None:
            raise ValueError("Condition is required for sampling")
        
        # Get device from the model parameters
        device = next(self.parameters()).device
        
        # Start from random noise
        x_t = torch.randn(shape, device=device)
        
        # Reverse diffusion process
        for t in range(self.T, 0, -1):
            # Sample noise (zero for the last step)
            noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            
            # Create timestep tensor
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Single step of reverse diffusion
            x_t = self.reverse_diffusion(x_t, t_tensor, noise, cond)
        
        return x_t
    
    def forward_training(self, cond):
        """
        Method for training with standard loss functions (like SmoothL1Loss).
        Unlike the forward method, this will create a computational graph for backpropagation.
        
        Args:
            cond: Conditioning data (xPDF) with shape [batch_size, 6000]
            
        Returns:
            Predicted atom positions with shape [batch_size, num_atoms, 3]
        """
        # Get batch size
        batch_size = cond.shape[0]
        
        # Define output shape for atomic positions
        shape = (batch_size, self.num_atoms, 3)
        
        # Get device from the model parameters
        device = next(self.parameters()).device
        
        # Get latent embedding from encoder
        latent_emb = self.encoder(cond)
        
        # Start from random noise
        x_t = torch.randn(shape, device=device)
        
        # Perform a single step of reverse diffusion from timestep 1
        # This will create a computational graph for backpropagation
        t_tensor = torch.ones((batch_size,), device=device, dtype=torch.long)
        t_normalized = t_tensor.float() / self.T
        
        # Predict the noise using the denoiser
        predicted_noise = self.denoiser(x_t, t_normalized.unsqueeze(1), latent_emb)
        
        # Use the predicted noise to get a cleaner sample
        # This is a simplified version of the reverse diffusion process,
        # but it will create a computational graph
        alpha_t = self.alphas[t_tensor].view(batch_size, 1, 1)
        beta_t = self.betas[t_tensor].view(batch_size, 1, 1)
        alpha_bar_t = self.alpha_bars[t_tensor].view(batch_size, 1, 1)
        
        # Calculate mean for the reverse process
        x_0_pred = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )
        
        return x_0_pred




