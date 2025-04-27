import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import os


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, cond_dim=6000, cond_embed_dim=64):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
          cond_dim: The dimensionality of the conditioning vector.
          cond_embed_dim: The dimensionality to embed the conditioning vector.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        
        # Condition embedding layers to reduce dimensionality of the conditioning vector
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, cond_embed_dim)
        )
        
        # Encoding layers where the resolution decreases
        # Input has 3 channels (RGB) + conditional channels
        self.conv1 = nn.Conv2d(3 + cond_embed_dim, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        
        # Track the output shapes after each convolution for later use in transpose convolutions
        # For 10x10 input:
        # After conv1: 10x10 (stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False) 
        # After conv2: 5x5 (stride=2, padding=1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        # After conv3: 3x3 (stride=2, padding=1) - because of rounding down on odd dimensions
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        # After conv4: 2x2 (stride=2, padding=1)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # Decoding layers where the resolution increases
        # Need to exactly match the encoder's dimensions
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        
        self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        
        # Output has 3 channels for coordinates
        self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, 3, 3, stride=1, padding=1)
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
    
    def forward(self, x, t, cond): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))  
        
        # Embedding of condition vector
        cond_embedded = self.cond_embed(cond)
        cond_embedded = cond_embedded.view(x.shape[0], cond_embedded.shape[1], 1, 1).expand(x.shape[0], cond_embedded.shape[1], x.shape[2], x.shape[3])
        
        net_input = torch.cat((x, cond_embedded), 1)  
        
        # Encoding path
        h1 = self.conv1(net_input)    
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        
        # Ensure h and h3 have the same spatial dimensions before concatenating
        if h.shape[2:] != h3.shape[2:]:
            h = F.interpolate(h, size=h3.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        
        # Ensure h and h2 have the same spatial dimensions before concatenating
        if h.shape[2:] != h2.shape[2:]:
            h = F.interpolate(h, size=h2.shape[2:], mode='bilinear', align_corners=False)
            
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        
        # Ensure h and h1 have the same spatial dimensions before concatenating
        if h.shape[2:] != h1.shape[2:]:
            h = F.interpolate(h, size=h1.shape[2:], mode='bilinear', align_corners=False)
            
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class DDPM(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2, cond_dim=6000, image_size=(10, 10), channels=3):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1 
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        cond_dim: int
            Dimensionality of the conditioning vector.
        image_size: tuple
            Size of the image (height, width).
        channels: int
            Number of color channels in the image.
        """
        
        super(DDPM, self).__init__()

        # Store image dimensions
        self.image_size = image_size
        self.channels = channels

        # Pass input directly to network, no reshaping needed
        self._network = network
        self.network = lambda x, t, cond: self._network(x, (t.squeeze()/T), cond)

        # Total number of time steps
        self.T = T
        self.cond_dim = cond_dim

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T+1))
        self.register_buffer("alpha", 1-self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))
        

    def forward_diffusion(self, x0, t, epsilon):
        '''
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon. 
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image) of shape [batch_size, channels, height, width]
        t: int
            step index 
        epsilon:
            noise sample of same shape as x0

        Returns
        -------
        torch.tensor
            image at timestep t, same shape as x0
        ''' 
        
        # Squeeze the time dimension to make it [batch_size]
        t_flat = t.squeeze(-1)
        
        # Now alpha_bar[t_flat] will have shape [batch_size]
        mean = torch.sqrt(self.alpha_bar[t_flat])[:, None, None, None] * x0
        std = torch.sqrt(1 - self.alpha_bar[t_flat])[:, None, None, None]
        
        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon, cond):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t of shape [batch_size, channels, height, width]
        t: int
            step index
        epsilon:
            noise sample of same shape as xt
        cond: torch.tensor
            Conditioning vector of shape [batch_size, cond_dim]

        Returns
        -------
        torch.tensor
            image at timestep t-1, same shape as xt
        """
        
        # Squeeze the time dimension
        t_flat = t.squeeze(-1)
        
        alpha_t = self.alpha[t_flat][:, None, None, None]
        beta_t = self.beta[t_flat][:, None, None, None]
        alpha_bar_t = self.alpha_bar[t_flat][:, None, None, None]
        
        # Handle t=0 case separately to avoid indexing errors
        if t_flat[0] > 0:
            alpha_bar_prev = self.alpha_bar[t_flat-1][:, None, None, None]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        
        # Predict the noise
        predicted_noise = self.network(xt, t, cond)
        
        # Calculate mean for p(x_{t-1} | x_t)
        mean = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        # Calculate variance for p(x_{t-1} | x_t)
        var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        std = torch.sqrt(var)
        
        return mean + std * epsilon

    
    @torch.no_grad()
    def sample(self, batch_size, cond=None):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        batch_size: int
            Number of samples to generate
        cond: torch.tensor or None
            Conditioning vector of shape [batch_size, cond_dim]
            If None, a random conditioning vector will be generated

        Returns
        -------
        torch.tensor
            sampled images of shape [batch_size, channels, height, width]            
        """
        # Check if conditioning vector has the right shape
        if cond is not None:
            # If cond is [1, cond_dim], reshape to [cond_dim]
            if cond.dim() == 2 and cond.shape[0] == 1:
                cond = cond.squeeze(0)
        if cond is None:
            # Generate random conditioning vector if none provided
            cond = torch.randn(batch_size, self.cond_dim, device=self.beta.device)
            print(f"Using randomly generated conditioning vector")
        elif cond.dim() == 1:
            # If single conditioning vector provided, expand to batch size
            cond = cond.unsqueeze(0).expand(batch_size, -1)
        
        # Sample xT: Gaussian noise of shape [batch_size, channels, height, width]
        xT = torch.randn(batch_size, self.channels, self.image_size[0], self.image_size[1], device=self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            # Sample noise for current step (or zero for t=1)
            noise = torch.randn_like(xt) if t > 1 else 0
            
            # Expand t to batch dimension with shape [batch_size, 1]
            t_batch = torch.tensor(t).expand(batch_size, 1).to(self.beta.device)
            
            # Single step of reverse diffusion
            xt = self.reverse_diffusion(xt, t_batch, noise, cond)

        return xt

    
    def elbo_simple(self, x0, cond):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image of shape [batch_size, channels, height, width]
        cond: torch.tensor
            Conditioning vector of shape [batch_size, cond_dim]

        Returns
        -------
        float
            ELBO value            
        """

        # Sample time step t with shape [batch_size, 1]
        t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)
        
        # Sample noise with same shape as x0
        epsilon = torch.randn_like(x0)

        # Forward diffusion to produce image at step t
        xt = self.forward_diffusion(x0, t, epsilon)
        
        # Predict noise and calculate loss
        predicted_noise = self.network(xt, t, cond)
        return -nn.MSELoss(reduction='mean')(epsilon, predicted_noise)

    
    def loss(self, x0, cond):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo_simple(x0, cond).mean()


def train(model, optimizer, scheduler, train_data, val_data=None, test_data=None, epochs=100, batch_size=256, device='cuda', ema=True, per_epoch_callback=None):
    """
    Training loop with validation
    
    Parameters
    ----------
    model: nn.Module
        Pytorch model
    optimizer: optim.Optimizer
        Pytorch optimizer to be used for training
    scheduler: optim.LRScheduler
        Pytorch learning rate scheduler
    train_data: Dataset or DataLoader
        Training data as a dataset or dataloader
    val_data: Dataset or DataLoader, optional
        Validation data for evaluating during training
    test_data: Dataset or DataLoader, optional
        Test data for final evaluation (not used in training loop)
    epochs: int
        Number of epochs to train
    batch_size: int
        Batch size for training
    device: torch.device
        Pytorch device specification
    ema: Boolean
        Whether to activate Exponential Model Averaging
    per_epoch_callback: function
        Called at the end of every epoch
    """


    # Create dataloader if dataset is provided
    if not isinstance(train_data, torch.utils.data.DataLoader):
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        dataloader = train_data

    # Setup validation dataloader if provided
    if val_data is not None and not isinstance(val_data, torch.utils.data.DataLoader):
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        val_dataloader = val_data

    # Setup progress bar
    total_steps = len(dataloader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(model, device=device, decay=1.0 - ema_alpha)                
    
    # Lists to track metrics
    train_losses = []
    val_maes = []
    
    for epoch in range(epochs):
        # Switch to train mode
        model.train()

        epoch_losses = []
        global_step_counter = 0
        for i, (x, cond) in enumerate(dataloader):
            x = x.to(device)
            cond = cond.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, cond)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter%ema_steps==0:
                    ema_model.update_parameters(model)                
        
        # Average training loss for this epoch
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation step
        if val_dataloader is not None:
            active_model = ema_model.module if ema else model
            val_mae = validate(active_model, val_dataloader, device)
            val_maes.append(val_mae)
            
            # Update progress bar with validation metric
            progress_bar.set_postfix(loss=f"⠀{avg_train_loss:12.4f}", val_mae=f"{val_mae:12.4f}", 
                                     epoch=f"{epoch+1}/{epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            
            print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model, epoch)
    
    # Return training metrics
    return {
        'train_losses': train_losses,
        'val_maes': val_maes
    }


def validate(model, dataloader, device):
    """
    Validate the model on validation data
    
    Parameters
    ----------
    model: nn.Module
        Trained DDPM model
    dataloader: DataLoader
        Validation dataloader
    device: torch.device
        Pytorch device specification
        
    Returns
    -------
    float
        Mean absolute error on validation data
    """
    from benchmark_tasks_utils import position_MAE
    
    model.eval()
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for x, cond in dataloader:
            x = x.to(device)
            cond = cond.to(device)
            
            # Generate samples using the model's sampling functionality
            samples = model.sample(x.size(0), cond)
            
            # Reshape to [batch_size, num_atoms, 3] for MAE calculation
            # Permute from [batch_size, channels, height, width] to [batch_size, height, width, channels]
            # Then reshape to [batch_size, height*width, channels]
            batch_size = x.size(0)
            
            # Permute dimensions before reshaping to correctly align channels
            pred_xyz = samples.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
            true_xyz = x.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
            
            all_preds.append(pred_xyz)
            all_truths.append(true_xyz)
    
    # Concatenate all predictions and ground truths
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_truths = torch.cat(all_truths, dim=0)
        
        # Calculate MAE
        mae = position_MAE(all_preds, all_truths)
        return mae.item()
    
    return float('inf')  # Return infinity if no validation data


def test(model, test_data, batch_size=256, device='cuda'):
    """
    Test the model on test data
    
    Parameters
    ----------
    model: nn.Module
        Trained DDPM model
    test_data: Dataset or DataLoader
        Test dataset or dataloader
    batch_size: int
        Batch size for testing
    device: torch.device
        Pytorch device specification
        
    Returns
    -------
    float
        Mean absolute error on test data
    """
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
    test_mae = validate(model, test_dataloader, device)
    print(f"Test MAE: {test_mae:.4f}")
    
    return test_mae


def sample_and_save_images(model, cond_vectors, num_samples=10, save_dir='samples'):
    """
    Sample images from the model with different conditioning vectors and save them to disk.
    
    Parameters
    ----------
    model: nn.Module
        Trained DDPM model
    cond_vectors: list of torch.tensor
        List of conditioning vectors to use for sampling
    num_samples: int
        Number of samples to generate per conditioning vector
    save_dir: str
        Directory to save generated samples
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Switch to eval mode
    model.eval()

    # Sample for each conditioning vector
    for i, cond in enumerate(cond_vectors):
        with torch.no_grad():
            # Generate samples directly in 3D format [batch_size, channels, height, width]
            samples = model.sample(num_samples, cond.to(model.beta.device)).cpu()
            
            # Map pixel values back from [-1,1] to [0,1]
            samples = (samples+1)/2 
            samples = samples.clamp(0.0, 1.0)

            # Create a grid of images
            grid = utils.make_grid(samples, nrow=int(math.sqrt(num_samples)))
            
            # Convert to PIL image and save
            img = transforms.functional.to_pil_image(grid)
            img.save(f"{save_dir}/cond_{i}.png")
            
            # Optionally display
            plt.figure(figsize=(12, 6))
            plt.imshow(transforms.functional.to_pil_image(grid))
            plt.axis('off')
            plt.title(f"Generated samples for condition vector {i}")
            plt.savefig(f"{save_dir}/cond_{i}_figure.png")
            plt.close()


class VectorConditionedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that pairs images with vector conditional information.
    Also includes atom types as an additional channel.
    """
    def __init__(self, data, model_type='unknown', atom_mapping_path=None):
        """
        Parameters
        ----------
        data: list
            List of data batches
        model_type: str
            Type of model ('pos_abs' or 'pos_frac')
        atom_mapping_path: str, optional
            Path to the atom type mapping JSON file. If provided, atom types will be included as a 4th channel.
        """
        self.images = []
        self.atom_types = []
        self.conditioning = []
        self.atom_mapping = None
        self.num_categories = 0
        
        # Load atom type mapping if provided
        if atom_mapping_path:
            try:
                import json
                with open(atom_mapping_path, 'r') as f:
                    self.atom_mapping = json.load(f)
                self.num_categories = self.atom_mapping['num_categories']
                print(f"Loaded atom mapping with {self.num_categories} categories")
            except Exception as e:
                print(f"Warning: Failed to load atom mapping: {e}")
                self.atom_mapping = None
        
        has_atom_types = self.atom_mapping is not None
        
        for batch in data:
            # Get position data (xyz coordinates)
            if model_type == 'pos_abs':
                pos = batch.pos_abs
            else:
                pos = batch.pos_frac
            
            # Reshape positions to [batch_size, 3, height, width]
            pos_reshaped = pos.view(-1, 3, 10, 10)
            self.images.append(pos_reshaped)
            
            # Process atom types separately if mapping is available
            if has_atom_types and hasattr(batch, 'x'):
                # Extract atom numbers (assuming the first column of x contains atom numbers)
                atom_numbers = batch.x[:, 0].cpu()
                
                # Map atom numbers to indices using the mapping
                atom_num_to_idx = self.atom_mapping['atom_num_to_idx']
                atom_indices = torch.zeros_like(atom_numbers)
                
                # Convert each atom number to its corresponding index
                for i, atom_num in enumerate(atom_numbers):
                    atom_num_int = int(atom_num.item())
                    # Default to 0 if atom type not in mapping
                    atom_indices[i] = int(atom_num_to_idx.get(str(atom_num_int), 0))
                
                # Keep atom indices as discrete values (no normalization)
                # Reshape to match position data shape
                atom_indices_reshaped = atom_indices.view(-1, 1, 10, 10)
                self.atom_types.append(atom_indices_reshaped)
            
            # Get conditioning vector (xPDF)
            xpdf = batch.y['xPDF']
            self.conditioning.append(xpdf[:,1,:])
        
        # Combine position data
        self.images = torch.cat(self.images, dim=0)
        
        # Combine atom types if available
        if has_atom_types:
            self.atom_types = torch.cat(self.atom_types, dim=0)
        
        # Combine conditioning data
        self.conditioning = torch.cat(self.conditioning, dim=0)
        self.conditioning = self.conditioning.squeeze(1)
        
        # Report dataset info
        print(f"Created dataset with {len(self.images)} samples, "
              f"image shape: {self.images.shape}, "
              f"conditioning shape: {self.conditioning.shape}")
        if has_atom_types:
            print(f"Atom types shape: {self.atom_types.shape}, "
                  f"with {self.num_categories} categories")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Return positions and atom types as separate tensors
        if len(self.atom_types) > 0:
            return (self.images[idx], self.atom_types[idx]), self.conditioning[idx]
        else:
            return self.images[idx], self.conditioning[idx]


def save_metrics_to_csv(metrics, filepath, model_params=None):
    """
    Save final training and validation metrics to a CSV file along with model parameters
    
    Parameters
    ----------
    metrics: dict
        Dictionary containing metrics to save
    filepath: str
        Path to save the CSV file
    model_params: dict
        Dictionary containing model parameters to save
    """
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
    
    if 'test_mae' in metrics:
        data['test_mae'] = metrics['test_mae']
    
    # Add model parameters if provided
    if model_params:
        data.update(model_params)
    
    # Convert to DataFrame (single row)
    df = pd.DataFrame([data])
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Final metrics saved to {filepath}")


def train_vector_conditioned_ddpm(train_data, val_data=None, test_data=None, T=1000, learning_rate=1e-3, 
                                  epochs=100, batch_size=256, ema=True, cond_dim=6000, 
                                  cond_embed_dim=64, image_size=(10, 10), model_type='unknown',
                                  atom_mapping_path=None):
    """
    Train a vector-conditioned DDPM model on RGB images
    
    Parameters
    ----------
    train_data: torch.utils.data.Dataset or list
        Training data to create dataset from
    val_data: torch.utils.data.Dataset or list, optional
        Validation data to create dataset from
    test_data: torch.utils.data.Dataset or list, optional
        Test data to create dataset from
    T: int
        Number of diffusion steps
    learning_rate: float
        Learning rate for optimizer
    epochs: int
        Number of training epochs
    batch_size: int
        Batch size for training
    ema: bool
        Whether to use exponential moving average
    cond_dim: int
        Dimensionality of the input conditioning vectors
    cond_embed_dim: int
        Dimensionality to embed the conditioning vectors to
    image_size: tuple
        Size of the images (height, width)
    model_type: str
        Type of model being trained (e.g., 'pos_abs' or 'pos_frac')
    atom_mapping_path: str, optional
        Path to the atom type mapping JSON file. If provided, atom types will be included as a 4th channel.
        
    Returns
    -------
    model: nn.Module
        Trained DDPM model
    metrics: dict
        Dictionary containing training metrics
    """
    # Create output directory for sample images
    import os
    
    # Determine number of channels based on whether atom mapping is provided
    channels = 4 if atom_mapping_path else 3
    
    # Create a unique folder for this model run based on parameters
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    atom_suffix = "_with_atoms" if atom_mapping_path else ""
    model_params = f"{model_type}_T{T}_lr{learning_rate}_epochs{epochs}_batch{batch_size}_cond{cond_embed_dim}{atom_suffix}_{timestamp}"
    samples_dir = os.path.join("training_samples", model_params)
    
    if not os.path.exists("training_samples"):
        os.makedirs("training_samples")
    
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Load atom type mapping if provided
    atom_mapping = None
    num_atom_categories = 0
    if atom_mapping_path:
        try:
            import json
            with open(atom_mapping_path, 'r') as f:
                atom_mapping = json.load(f)
            num_atom_categories = atom_mapping['num_categories']
            print(f"Loaded atom mapping with {num_atom_categories} categories for visualization")
        except Exception as e:
            print(f"Warning: Could not load atom mapping for visualization: {e}")
    
    # Create custom datasets that pair images with conditioning vectors
    train_dataset = VectorConditionedDataset(train_data, model_type=model_type, atom_mapping_path=atom_mapping_path) if not isinstance(train_data, torch.utils.data.Dataset) else train_data
    val_dataset = VectorConditionedDataset(val_data, model_type=model_type, atom_mapping_path=atom_mapping_path) if val_data is not None and not isinstance(val_data, torch.utils.data.Dataset) else val_data
    test_dataset = VectorConditionedDataset(test_data, model_type=model_type, atom_mapping_path=atom_mapping_path) if test_data is not None and not isinstance(test_data, torch.utils.data.Dataset) else test_data

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Construct Unet
    unet = ScoreNet(
        (lambda t: torch.ones(1).to(device)),
        cond_dim=cond_dim,
        cond_embed_dim=cond_embed_dim
    )

    # Construct model
    model = DDPM(
        unet, 
        T=T, 
        cond_dim=cond_dim, 
        image_size=image_size, 
        channels=channels
    ).to(device)

    # Construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

    # Save reference to first few images for sampling comparison
    # Get the first batch from the train dataset for visualization
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,  # Only need a small batch for visualization
        shuffle=True
    )
    
    # Initialize storage for ground truth data
    ground_truth_images = []
    ground_truth_atom_types = []
    ground_truth_conds = []
    
    # Get reference data for visualization
    for batch in train_dataloader:
        # Check if this is the new format with atom types ((positions, atom_types), cond)
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            (positions, atom_types), conds = batch
            ground_truth_images = positions[:2].to(device)
            ground_truth_atom_types = atom_types[:2].to(device)
            ground_truth_conds = conds[:2].to(device)
        else:
            # Traditional format (pos, cond)
            positions, conds = batch
            ground_truth_images = positions[:2].to(device)
            ground_truth_conds = conds[:2].to(device)
            # Empty atom types
            ground_truth_atom_types = []
        break  # Only need the first batch

    def reporter(model, epoch):
        """Callback function used for plotting images during training after each epoch"""
        # Switch to eval mode
        model.eval()

        with torch.no_grad():
            # Create a list to store all rows (each row is gt + 3 samples)
            all_rows = []
            
            # Store all 3D sample points for later plotting
            all_gt_points = []
            all_sample_points = []
            all_gt_atom_types = []
            all_sample_atom_types = []
            
            # Check if we have atom types data
            have_atom_types = len(ground_truth_atom_types) > 0
            
            # For each ground truth image
            for gt_idx in range(len(ground_truth_images)):
                # Get ground truth image and conditioning vector
                gt_image = ground_truth_images[gt_idx:gt_idx+1]
                cond = ground_truth_conds[gt_idx:gt_idx+1]
                
                # Get ground truth atom type if available
                gt_atom_type = None
                if have_atom_types:
                    gt_atom_type = ground_truth_atom_types[gt_idx:gt_idx+1]
                
                # Generate 3 samples for each ground truth image
                samples = model.sample(3, cond).cpu()
                
                # Use raw data for visualization
                gt_normalized_img = gt_image.cpu()
                samples_normalized_img = samples.cpu()

                # Combine ground truth as first image followed by 3 samples (1 row)
                row = torch.cat([gt_normalized_img, samples_normalized_img], dim=0)
                all_rows.append(row)
                
                # Extract ground truth points for 3D plotting
                gt_points = gt_normalized_img[0, :3].permute(1, 2, 0)  # [height, width, channels]
                gt_points = gt_points.reshape(-1, 3)  # [height*width, channels] = [100, 3]
                all_gt_points.append(gt_points)
                
                # Extract sample points for 3D plotting
                row_samples = []
                row_atom_types = []
                for i in range(3):
                    sample_points = samples_normalized_img[i, :3].permute(1, 2, 0)
                    sample_points = sample_points.reshape(-1, 3)
                    row_samples.append(sample_points)
                    
                    # Extract atom types if available (4th channel)
                    if channels == 4:
                        sample_atom_type = samples[i, 3].view(-1)
                        row_atom_types.append(sample_atom_type)
                
                all_sample_points.append(row_samples)
                
                # Store ground truth atom types if available
                if have_atom_types:
                    gt_atom = gt_atom_type[0].cpu().view(-1)
                    all_gt_atom_types.append(gt_atom)
                    if row_atom_types:
                        all_sample_atom_types.append(row_atom_types)
            
            # 1. Create the 2D image grid visualization (using only coordinate channels)
            combined = torch.cat(all_rows, dim=0)
            grid = utils.make_grid(combined, nrow=4)
            plt.figure(figsize=(12, 3 * len(ground_truth_images)))
            plt.gca().set_axis_off()
            plt.imshow(transforms.functional.to_pil_image(grid))
            plt.title(f"Epoch {epoch} - Ground Truth (first column) + Generated Samples")
            
            # Save the 2D image grid
            filename = os.path.join(samples_dir, f"epoch_{epoch:03d}_all_images.png")
            plt.savefig(filename)
            plt.close()
            
            # 2. Create the 3D plot with 2 rows, each with GT + 3 samples
            fig = plt.figure(figsize=(20, 10), facecolor='white')
            
            # Custom function to style each 3D plot
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
                # Don't restrict the axes limits anymore
                # Remove tick labels for cleaner look
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            
            # Plot both rows (one for each ground truth)
            for row_idx in range(len(all_gt_points)):
                # Plot ground truth as first plot in each row
                plot_position = row_idx * 4 + 1  # 1 or 5
                ax_gt = fig.add_subplot(2, 4, plot_position, projection='3d')
                
                # If atom types are available, use them for coloring
                if have_atom_types:
                    # Use integer categories directly for coloring (no normalization)
                    colors = all_gt_atom_types[row_idx].numpy()
                    scatter = ax_gt.scatter(all_gt_points[row_idx][:, 0], 
                                          all_gt_points[row_idx][:, 1], 
                                          all_gt_points[row_idx][:, 2], 
                                          c=colors, cmap='tab10', marker='o', s=25, alpha=0.8)
                    
                    # Add color bar for the first row
                    if row_idx == 0 and num_atom_categories > 0:
                        cbar = plt.colorbar(scatter, ax=ax_gt, ticks=range(num_atom_categories))
                        if atom_mapping:
                            # Try to add atom labels if available
                            idx_to_atom_num = atom_mapping.get('idx_to_atom_num', {})
                            if idx_to_atom_num:
                                # Only show a subset of labels if there are many categories
                                if num_atom_categories <= 10:
                                    labels = [f"Z={idx_to_atom_num[str(i)]}" for i in range(num_atom_categories)]
                                    cbar.set_ticklabels(labels)
                else:
                    ax_gt.scatter(all_gt_points[row_idx][:, 0], all_gt_points[row_idx][:, 1], all_gt_points[row_idx][:, 2], 
                                 c='blue', marker='o', s=25, alpha=0.8)
                
                style_3d_axes(ax_gt, f'Ground Truth Structure')
                
                # Plot 3 samples for this ground truth
                for sample_idx in range(3):
                    plot_position = row_idx * 4 + sample_idx + 2  # [2,3,4] or [6,7,8]
                    ax = fig.add_subplot(2, 4, plot_position, projection='3d')
                    sample_points = all_sample_points[row_idx][sample_idx]
                    
                    # If atom types are available, use them for coloring
                    if have_atom_types and all_sample_atom_types:
                        # Use integer categories directly (no normalization)
                        colors = all_sample_atom_types[row_idx][sample_idx].numpy()
                        ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                                  c=colors, cmap='tab10', marker='o', s=25, alpha=0.8)
                    else:
                        ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                                  c='blue', marker='o', s=25, alpha=0.8)
                    
                    style_3d_axes(ax, f'Sample {sample_idx+1}')
            
            plt.tight_layout()
            
            # Save the 3D plot
            filename_3d = os.path.join(samples_dir, f"epoch_{epoch:03d}_3d_plot.png")
            plt.savefig(filename_3d)
            plt.close()

    # Call training loop with validation
    metrics = train(
        model, 
        optimizer, 
        scheduler, 
        train_dataset, 
        val_data=val_dataset,
        test_data=test_dataset,
        epochs=epochs, 
        batch_size=batch_size,
        device=device, 
        ema=ema, 
        per_epoch_callback=reporter
    )
    
    # Save model
    model_path = os.path.join(samples_dir, "vector_conditioned_rgb_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to '{model_path}'")
    
    # If test data is provided, run final evaluation
    if test_dataset is not None:
        test_mae = test(model, test_dataset, batch_size=batch_size, device=device)
        metrics['test_mae'] = test_mae
        print(f"Final test MAE: {test_mae:.4f}")
    
    # Collect model parameters
    model_parameters = {
        'model_type': model_type,
        'T': T,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'ema': ema,
        'cond_dim': cond_dim,
        'cond_embed_dim': cond_embed_dim,
        'image_size_h': image_size[0],
        'image_size_w': image_size[1],
        'channels': channels,
        'atom_mapping_path': atom_mapping_path,
        'num_atom_categories': num_atom_categories,
        'device': str(device)
    }
    
    # Save metrics to CSV
    metrics_path = os.path.join(samples_dir, "final_metrics.csv")
    save_metrics_to_csv(metrics, metrics_path, model_parameters)
    
    return model, metrics 