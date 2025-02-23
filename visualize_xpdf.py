import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_xpdf(xpdf_data, r_range=(0, 100), title="X-ray Pair Distribution Function", save_path=None):
    """
    Plot the X-ray Pair Distribution Function (xPDF).
    
    Args:
        xpdf_data (torch.Tensor): Tensor of shape (1, 2, N) where:
            - First dimension is batch
            - Second dimension contains [r, G(r)]
            - Third dimension is the number of points
        r_range (tuple): Range of r values to plot (min, max)
        title (str): Title of the plot
        save_path (str, optional): Path to save the figure. If None, displays the plot instead.
    """
    # Extract r and G(r) values from tensor
    r = xpdf_data[0, 0].numpy()    # First channel is r values
    g_r = xpdf_data[0, 1].numpy()  # Second channel is G(r) values
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the xPDF
    plt.plot(r, g_r, 'b-', linewidth=1.5, label='G(r)')
    
    # Set the r-range
    plt.xlim(r_range)
    
    # Add labels and title
    plt.xlabel('r (Å)', fontsize=12)
    plt.ylabel('G(r) (Å⁻²)', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_multiple_xpdfs(xpdf_list, labels=None, r_range=(0, 100), title="Comparison of xPDFs", save_path=None):
    """
    Plot multiple X-ray Pair Distribution Functions (xPDFs) for comparison.
    
    Args:
        xpdf_list (list): List of tensors, each of shape (1, 2, N)
        labels (list): List of labels for each xPDF
        r_range (tuple): Range of r values to plot (min, max)
        title (str): Title of the plot
        save_path (str, optional): Path to save the figure. If None, displays the plot instead.
    """
    plt.figure(figsize=(12, 7))
    
    if labels is None:
        labels = [f'xPDF {i+1}' for i in range(len(xpdf_list))]
    
    for xpdf, label in zip(xpdf_list, labels):
        r = xpdf[0, 0].numpy()    # First channel is r values
        g_r = xpdf[0, 1].numpy()  # Second channel is G(r) values
        plt.plot(r, g_r, linewidth=1.5, label=label)
    
    plt.xlim(r_range)
    plt.xlabel('r (Å)', fontsize=12)
    plt.ylabel('G(r) (Å⁻²)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 