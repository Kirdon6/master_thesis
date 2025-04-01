import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Any, Optional, List, Union
import os
from vector_diff import VectorDiffusion  # Import VectorDiffusion


class Reporter:
    """
    Reporter class for visualizing diffusion processes during training.
    
    Periodically takes a structure from the dataset and visualizes:
    1. Forward diffusion process (original structure to noise)
    2. Reverse diffusion process (random noise to predicted structure)
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        device: torch.device,
        save_dir: str
    ):
        """
        Initialize the Reporter.
        
        Args:
            config: Dictionary containing reporter configuration
            device: Device to run computations on
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        # Create samples directory for saving visualizations
        self.samples_dir = f"{save_dir}/samples"
        os.makedirs(self.samples_dir, exist_ok=True)
        self.device = device
        
        # Get configuration parameters with defaults
        reporter_config = config.get("Reporter_config", {})
        self.visualization_period = reporter_config.get("visualization_period", 10)
        self.timesteps_to_visualize = reporter_config.get("timesteps_to_visualize", [0, 25, 50, 75, 99])
        self.num_structures = reporter_config.get("num_structures", 1)
        self.figsize = reporter_config.get("figsize", (20, 10))  # Increased default figure size
        
        # Ensure timesteps are sorted
        self.timesteps_to_visualize = sorted(self.timesteps_to_visualize)
    
    def should_visualize(self, epoch: int) -> bool:
        """Check if visualization should be performed at the current epoch."""
        return epoch % self.visualization_period == 0
    
    def visualize_diffusion(
        self, 
        epoch: int, 
        model: VectorDiffusion, 
        sample_positions: torch.Tensor,
        sample_xPDF: torch.Tensor,
        diffusion_steps: int = 100
    ) -> None:
        """
        Visualize forward and reverse diffusion processes.
        
        Args:
            epoch: Current training epoch
            model: Diffusion model
            sample_positions: Tensor containing positions of atoms in a structure
            sample_xPDF: Tensor containing xPDF data
            diffusion_steps: Total number of diffusion steps
        """

        
        model.eval()
        with torch.no_grad():
            
            # Forward diffusion visualization (clean → noisy)
            forward_positions = []
            
            # Start with the original clean positions
            forward_positions.append(sample_positions.clone())
            
            # For each timestep to visualize, apply forward diffusion
            for t in self.timesteps_to_visualize[1:]:  # Skip t=0 which is already added
                # Generate random noise
                epsilon = torch.randn_like(sample_positions)
                # Apply forward diffusion
                noisy_pos = model.forward_diffusion(sample_positions, t, epsilon)
                forward_positions.append(noisy_pos)
            
            # Reverse diffusion visualization (noisy → clean)
            reverse_positions = []
            
            # Start with random noise
            x_t = forward_positions[-1]
            reverse_positions.append(x_t.clone())
            
            # For reverse visualization, we'll pick specific timesteps from T to 0
            reverse_timesteps = sorted(self.timesteps_to_visualize[1:], reverse=True) + [0]
            

            for t in range(diffusion_steps-1, -1, -1):
                if t in reverse_timesteps:
                    reverse_positions.append(x_t.clone())
                x_t = model.reverse_diffusion(x_t, torch.tensor([t], device=self.device), torch.randn_like(x_t, device=self.device), sample_xPDF.to(self.device))
            
            
            # Create visualization
            model.train()
            self._create_visualization(epoch, sample_positions, forward_positions, reverse_positions)

    def _create_visualization(
        self, 
        epoch: int, 
        original_pos: torch.Tensor, 
        forward_positions: List[torch.Tensor],
        reverse_positions: List[torch.Tensor]
    ) -> None:
        """Create and save visualization of diffusion processes."""
        num_timesteps = len(self.timesteps_to_visualize)
        
        # Create figure with 3D subplots with more space between subplots
        fig = plt.figure(figsize=self.figsize)
        fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Add more space between subplots
        
        # Set main title
        fig.suptitle(f"Diffusion Process Visualization (Epoch {epoch})", fontsize=16)
        
        # Create 3D axes for each subplot
        axes = []
        for i in range(2):  # 2 rows
            row_axes = []
            for j in range(num_timesteps):  # columns
                ax = fig.add_subplot(2, num_timesteps, i*num_timesteps + j + 1, projection='3d')
                # Reduce padding by adjusting the position
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height * 0.9])
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Plot original structure in the first cell
        self._plot_structure(axes[0][0], original_pos.cpu(), "Original")
        
        # Plot forward diffusion steps
        for i, pos in enumerate(forward_positions):
            if i == 0:  # Skip plotting original again
                continue
            title = f"Forward t={self.timesteps_to_visualize[i]}"
            self._plot_structure(axes[0][i], pos.cpu(), title)
        
        # Plot reverse diffusion steps (in reverse order - from noise to clean)
        reverse_timesteps = self.timesteps_to_visualize[::-1]  # Reverse the timesteps
        
        for i, pos in enumerate(reverse_positions):
            # Calculate the column index for reverse order display
            col_idx = num_timesteps - 1 - i  # Reverse the column order
            
            # Ensure we're within bounds
            if col_idx >= 0:
                title = f"Reverse t={reverse_timesteps[i]}"
                self._plot_structure(axes[1][col_idx], pos.cpu(), title)
        
        # Save figure without using tight_layout
        plt.savefig(f"{self.samples_dir}/diffusion_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_structure(self, ax, positions: torch.Tensor, title: str) -> None:
        """Plot 3D structure of positions on the given axis."""
        # Convert to numpy for plotting
        pos_np = positions.numpy()

        # Reshape to ensure [n_atoms, 3] format
        pos_np = pos_np.reshape(-1, 3)
        
        # Plot 3D scatter
        ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], c='b', alpha=0.7, s=20)
        
        # Set equal aspect ratio
        max_range = np.array([
            pos_np[:, 0].max() - pos_np[:, 0].min(),
            pos_np[:, 1].max() - pos_np[:, 1].min(),
            pos_np[:, 2].max() - pos_np[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (pos_np[:, 0].max() + pos_np[:, 0].min()) * 0.5
        mid_y = (pos_np[:, 1].max() + pos_np[:, 1].min()) * 0.5
        mid_z = (pos_np[:, 2].max() + pos_np[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Simplify axes appearance 
        ax.set_title(title, fontsize=10)
        
        # Remove axis ticks and labels for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Remove axis panes and spines for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w') 