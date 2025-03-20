import os
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.visualize.plot import plot_atoms
import argparse
from typing import List, Union, Optional, Tuple
import plotly.graph_objects as go

# Optional imports for more visualization options
try:
    import py3Dmol
    from IPython.display import display
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

def get_atomic_radius(atomic_number: int) -> float:
    """Get the atomic radius for a given atomic number."""
    # Simplified atomic radii in Angstroms
    radii = {
        1: 0.38,  # H
        6: 0.77,  # C
        7: 0.75,  # N
        8: 0.73,  # O
        11: 1.02, # Na
        12: 1.10, # Mg
        13: 1.18, # Al
        14: 1.11, # Si
        15: 1.06, # P
        16: 1.02, # S
        17: 0.99, # Cl
        19: 1.38, # K
        20: 1.00, # Ca
        22: 1.32, # Ti
        24: 1.18, # Cr
        25: 1.17, # Mn
        26: 1.17, # Fe
        27: 1.16, # Co
        28: 1.15, # Ni
        29: 1.17, # Cu
        30: 1.25, # Zn
        33: 1.14, # As
        34: 1.03, # Se
        35: 1.20, # Br
        38: 1.32, # Sr
        42: 1.30, # Mo
        47: 1.34, # Ag
        48: 1.48, # Cd
        50: 1.40, # Sn
        53: 1.40, # I
        56: 1.34, # Ba
        79: 1.34, # Au
        82: 1.46, # Pb
    }
    # Default radius if atomic number not in dictionary
    return radii.get(atomic_number, 1.0)

def get_element_name(atomic_number: int) -> str:
    """Get the element name for a given atomic number."""
    elements = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn'
    }
    return elements.get(atomic_number, f"Element{atomic_number}")

def get_element_color(atomic_number: int) -> Tuple[float, float, float]:
    """Get the color for a given atomic number."""
    # Common element colors used in molecular visualization
    colors = {
        1: (1.0, 1.0, 1.0),      # H: white
        6: (0.5, 0.5, 0.5),      # C: gray
        7: (0.0, 0.0, 1.0),      # N: blue
        8: (1.0, 0.0, 0.0),      # O: red
        11: (0.67, 0.36, 0.95),  # Na: purple
        12: (0.54, 1.0, 0.0),    # Mg: light green
        13: (0.75, 0.65, 0.65),  # Al: light gray
        14: (0.94, 0.78, 0.63),  # Si: beige
        15: (1.0, 0.5, 0.0),     # P: orange
        16: (1.0, 0.78, 0.16),   # S: yellow
        17: (0.12, 0.94, 0.12),  # Cl: green
        19: (0.56, 0.25, 0.83),  # K: purple
        20: (0.24, 1.0, 0.0),    # Ca: light green
        26: (0.88, 0.4, 0.2),    # Fe: brown
        29: (0.78, 0.5, 0.2),    # Cu: bronze
        47: (0.75, 0.75, 0.75),  # Ag: silver
        79: (1.0, 0.82, 0.14)    # Au: gold
    }
    
    # Default to light gray if atomic number is not found
    return colors.get(atomic_number, (0.8, 0.8, 0.8))

def load_structure(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Load a structure from a PyTorch file.
    
    Args:
        file_path: Path to the structure file
        
    Returns:
        positions: Atom positions
        node_features: Node features (including atomic numbers)
        edge_index: Edge connections
        metadata: Dictionary with structure metadata
    """
    data = torch.load(file_path, weights_only=False)
    
    # Extract relevant information
    positions = data.pos_abs  # Absolute positions
    node_features = data.x    # Node features (first column is atomic number)
    edge_index = data.edge_index  # Edge connections
    
    # Extract metadata
    metadata = {
        'data_id': data.data_id,
        'crystal_type': data.y['crystal_type'],
        'space_group': data.y['space_group_symbol'],
        'space_group_number': data.y['space_group_number'],
        'crystal_system': data.y['crystal_system'],
        'np_size': data.y['np_size']
    }
    
    return positions, node_features, edge_index, metadata

def load_positions_only(positions_tensor: torch.Tensor, default_atom_type: int = 14, 
                      bond_threshold: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Create visualization data from just positions tensor.
    
    Args:
        positions_tensor: Tensor of shape (n_atoms, 3) containing atom positions
        default_atom_type: Atomic number to use for all atoms (default: 14 = Silicon)
        bond_threshold: Distance threshold for creating bonds between atoms
        
    Returns:
        positions: Atom positions
        node_features: Generated node features (all same atom type)
        edge_index: Generated edge connections based on distance threshold
        metadata: Minimal metadata dictionary
    """
    # Ensure positions is a tensor with the right shape
    if isinstance(positions_tensor, np.ndarray):
        positions = torch.tensor(positions_tensor, dtype=torch.float32)
    else:
        positions = positions_tensor.clone()
    
    if len(positions.shape) == 1:
        # If positions is a flattened vector, reshape it to (n_atoms, 3)
        if positions.shape[0] % 3 == 0:
            n_atoms = positions.shape[0] // 3
            positions = positions.reshape(n_atoms, 3)
        else:
            raise ValueError(f"Positions tensor shape {positions.shape} cannot be reshaped to (n_atoms, 3)")
    
    # Create node features (all atoms same type)
    n_atoms = positions.shape[0]
    node_features = torch.ones((n_atoms, 1), dtype=torch.float32) * default_atom_type
    
    # Generate edge_index based on distance threshold
    edge_list = []
    
    # Compute pairwise distances and create edges for atoms within threshold
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = torch.norm(positions[i] - positions[j])
            if dist < bond_threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add both directions for undirected graph
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # If no edges were found, create empty edge index tensor
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Create basic metadata
    metadata = {
        'data_id': 'model_prediction',
        'crystal_type': 'Unknown',
        'space_group': 'Unknown',
        'space_group_number': 0,
        'crystal_system': 'Unknown',
        'np_size': 0.0
    }
    
    return positions, node_features, edge_index, metadata

def plot_structure_matplotlib(positions: torch.Tensor, node_features: torch.Tensor, 
                             edge_index: torch.Tensor, metadata: dict, 
                             output_path: Optional[str] = None,
                             show_edges: bool = True, fig_size: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize the structure using matplotlib.
    
    Args:
        positions: Atom positions
        node_features: Node features (including atomic numbers)
        edge_index: Edge connections
        metadata: Dictionary with structure metadata
        output_path: Optional path to save the figure
        show_edges: Whether to show edges between atoms
        fig_size: Figure size
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get atomic numbers (first column of node_features)
    atomic_numbers = node_features[:, 0].int().numpy()
    
    # Plot atoms
    for i, (pos, atom_num) in enumerate(zip(positions, atomic_numbers)):
        x, y, z = pos.numpy()
        radius = get_atomic_radius(atom_num.item()) * 10  # Scale radius for visualization
        color = get_element_color(atom_num.item())
        element = get_element_name(atom_num.item())
        
        ax.scatter(x, y, z, color=color, s=radius, alpha=0.7, label=f"{element}" if i == 0 else "")
    
    # Plot edges if requested
    if show_edges:
        for i in range(edge_index.shape[1]):
            idx1, idx2 = edge_index[:, i].numpy()
            x1, y1, z1 = positions[idx1].numpy()
            x2, y2, z2 = positions[idx2].numpy()
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='black', alpha=0.3, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f"Structure ID: {metadata['data_id']}\n"
                f"Crystal: {metadata['crystal_system']} ({metadata['space_group']})")
    
    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add a legend showing unique elements
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])
            
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()

def plot_structure_ase(positions: torch.Tensor, node_features: torch.Tensor, 
                      metadata: dict, output_path: Optional[str] = None,
                      rotation: str = '45x,45y,45z', fig_size: Tuple[int, int] = (10, 8),
                      edge_index: Optional[torch.Tensor] = None, show_edges: bool = True) -> None:
    """
    Visualize the structure using ASE's plotting tools.
    
    Args:
        positions: Atom positions
        node_features: Node features (including atomic numbers)
        metadata: Dictionary with structure metadata
        output_path: Optional path to save the figure
        rotation: Rotation string for the view
        fig_size: Figure size
        edge_index: Edge connections
        show_edges: Whether to show edges between atoms
    """
    # Get atomic numbers (first column of node_features)
    atomic_numbers = node_features[:, 0].int().numpy()
    
    # Create ASE Atoms object
    atoms = Atoms(symbols=atomic_numbers, positions=positions.numpy())
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot atoms with smaller radii scaling
    plot_atoms(atoms, ax, rotation=rotation, show_unit_cell=False, radii=0.3, scale=0.5)
    
    # Add bonds manually if requested
    if show_edges and edge_index is not None:
        # Get the transformed positions after rotation
        # We need to extract this from the existing plot since ASE applies the rotation
        scatter_collection = None
        for child in ax.get_children():
            if isinstance(child, plt.matplotlib.collections.PathCollection):
                scatter_collection = child
                break
        
        if scatter_collection is not None:
            # Get the transformed positions
            transformed_positions = scatter_collection.get_offsets()
            
            # Draw bonds
            seen_pairs = set()
            for i in range(edge_index.shape[1]):
                idx1, idx2 = edge_index[:, i].numpy()
                pair = tuple(sorted([idx1, idx2]))
                
                if pair not in seen_pairs:
                    pos1 = transformed_positions[idx1]
                    pos2 = transformed_positions[idx2]
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           color='gray', alpha=0.5, linewidth=1.0)
                    seen_pairs.add(pair)
    
    # Set title
    ax.set_title(f"Structure ID: {metadata['data_id']}\n"
                f"Crystal: {metadata['crystal_system']} ({metadata['space_group']})")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()

def plot_structure_plotly(positions: torch.Tensor, node_features: torch.Tensor, 
                        edge_index: torch.Tensor, metadata: dict, 
                        output_path: Optional[str] = None, 
                        show_edges: bool = True) -> None:
    """
    Visualize the structure using Plotly for interactive 3D visualization.
    
    Args:
        positions: Atom positions
        node_features: Node features (including atomic numbers)
        edge_index: Edge connections
        metadata: Dictionary with structure metadata
        output_path: Optional path to save the figure
        show_edges: Whether to show edges between atoms
    """
    # Convert to numpy for easier handling
    pos = positions.numpy()
    atomic_numbers = node_features[:, 0].int().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Add atoms as markers
    atom_trace = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode='markers',
        marker=dict(
            size=[get_atomic_radius(an.item()) * 15 for an in atomic_numbers],
            color=[f'rgb({r*255},{g*255},{b*255})' for an in atomic_numbers 
                  for r, g, b in [get_element_color(an.item())]],
            line=dict(width=0.5, color='rgb(50,50,50)')
        ),
        text=[f"{get_element_name(an.item())}" for an in atomic_numbers],
        hoverinfo='text',
        name='Atoms'
    )
    
    fig.add_trace(atom_trace)
    
    # Add edges if requested
    if show_edges and edge_index is not None:
        edge_x = []
        edge_y = []
        edge_z = []
        
        for i in range(edge_index.shape[1]):
            idx1, idx2 = edge_index[:, i].numpy()
            x1, y1, z1 = pos[idx1]
            x2, y2, z2 = pos[idx2]
            
            # Add line coordinates
            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])
            edge_z.extend([z1, z2, None])
        
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='rgba(70,70,70,0.8)', width=3),  # Increased width and opacity
            hoverinfo='none',
            name='Bonds'
        )
        
        fig.add_trace(edge_trace)
    
    # Set layout
    title = f"Structure ID: {metadata['data_id']}"
    if metadata['crystal_system'] != 'Unknown':
        title += f" - Crystal: {metadata['crystal_system']} ({metadata['space_group']})"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (Å)'),
            yaxis=dict(title='Y (Å)'),
            zaxis=dict(title='Z (Å)'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=1),
        template='plotly_white'
    )
    
    # Save figure if output path provided
    if output_path:
        fig.write_html(output_path)
        print(f"Interactive figure saved to {output_path}")
    
    # Show figure
    fig.show()

def plot_structure_py3dmol(positions: torch.Tensor, node_features: torch.Tensor, 
                          edge_index: torch.Tensor, metadata: dict) -> None:
    """
    Visualize the structure using py3Dmol (requires py3Dmol package and Jupyter environment).
    
    Args:
        positions: Atom positions
        node_features: Node features (including atomic numbers)
        edge_index: Edge connections
        metadata: Dictionary with structure metadata
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol is not available. Please install it with: pip install py3Dmol")
        return
    
    # Create a py3Dmol view
    view = py3Dmol.view(width=800, height=600)
    
    # Convert positions and atomic numbers to appropriate format
    pos = positions.numpy()
    atomic_numbers = node_features[:, 0].int().numpy()
    
    # Add atoms to the view
    for i, (p, an) in enumerate(zip(pos, atomic_numbers)):
        element = get_element_name(an.item())
        color = get_element_color(an.item())
        
        # Convert color from (r,g,b) format to hex
        hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        
        # Add atom as sphere
        view.addSphere({
            'center': {'x': p[0], 'y': p[1], 'z': p[2]},
            'radius': get_atomic_radius(an.item()) * 0.5,
            'color': hex_color,
            'alpha': 0.9
        })
    
    # Add bonds (edges)
    for i in range(edge_index.shape[1]):
        idx1, idx2 = edge_index[:, i].numpy()
        p1 = pos[idx1]
        p2 = pos[idx2]
        
        view.addCylinder({
            'start': {'x': p1[0], 'y': p1[1], 'z': p1[2]},
            'end': {'x': p2[0], 'y': p2[1], 'z': p2[2]},
            'radius': 0.1,
            'color': 'gray',
            'fromCap': True,
            'toCap': True,
            'alpha': 0.5
        })
    
    # Set title
    title = f"Structure ID: {metadata['data_id']}"
    if metadata['crystal_system'] != 'Unknown':
        title += f" - Crystal: {metadata['crystal_system']} ({metadata['space_group']})"
    
    view.addLabel(title, {'position': {'x': 0, 'y': 0, 'z': 0}, 'backgroundColor': 'white', 'fontColor': 'black'})
    
    # Set view options
    view.zoomTo()
    view.setStyle({'sphere': {}})
    
    # Display the view
    view.show()

def read_dataset_indices(file_path: str) -> List[int]:
    """
    Read dataset indices from a CSV file.
    
    Args:
        file_path: Path to the CSV file with indices
        
    Returns:
        List of dataset indices
    """
    indices = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    indices.append(int(line))
                except ValueError:
                    # Skip if not a valid integer
                    continue
    return indices

def visualize_structure(
    positions_data: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
    data_dir: str = 'data/CHILI-3K/processed_central',
    dataset_file: str = 'data/CHILI-3K/datasplit_random_central_test.csv',
    index: Optional[int] = None,
    method: str = 'matplotlib',
    output_path: Optional[str] = None,
    show_edges: bool = True,
    default_atom: int = 14,
    bond_threshold: float = 3.0
) -> None:
    """
    Visualize atomic structure from either position data or CHILI dataset.
    
    Args:
        positions_data: Position vectors as tensor/array or path to file containing positions
        data_dir: Directory containing processed structure data files
        dataset_file: File containing dataset indices
        index: Specific index to visualize (if not specified, a random one is chosen)
        method: Visualization method ('matplotlib', 'ase', 'plotly', 'py3dmol')
        output_path: Path to save the output figure
        show_edges: Show edges/bonds between atoms
        default_atom: Default atom type when visualizing just position vectors
        bond_threshold: Distance threshold for creating bonds
    """
    # Handle positions data if provided
    if positions_data is not None:
        if isinstance(positions_data, str):
            print(f"Loading positions from {positions_data}")
            try:
                # Try to load as PyTorch tensor
                data = torch.load(positions_data)
                
                # If data is a dictionary or similar, try to extract positions
                if hasattr(data, 'get') and callable(data.get):
                    if 'positions' in data:
                        data = data['positions']
                    elif 'pos' in data:
                        data = data['pos']
                    elif 'coords' in data:
                        data = data['coords']
                
                positions, node_features, edge_index, metadata = load_positions_only(
                    data, 
                    default_atom_type=default_atom,
                    bond_threshold=bond_threshold
                )
            except Exception as e:
                print(f"Error loading positions file: {e}")
                try:
                    # Try to load as NumPy array
                    data = np.load(positions_data)
                    positions, node_features, edge_index, metadata = load_positions_only(
                        data, 
                        default_atom_type=default_atom,
                        bond_threshold=bond_threshold
                    )
                except Exception as e2:
                    print(f"Could not load positions file as NumPy array either: {e2}")
                    return
        else:
            # Direct tensor/array input
            positions, node_features, edge_index, metadata = load_positions_only(
                positions_data,
                default_atom_type=default_atom,
                bond_threshold=bond_threshold
            )
    else:
        # Load from CHILI dataset
        if os.path.exists(dataset_file):
            indices = read_dataset_indices(dataset_file)
        else:
            print(f"Dataset file {dataset_file} not found. Listing all available structures.")
            indices = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(data_dir) 
                    if f.startswith('data_') and f.endswith('.pt')]
        
        # Select an index to visualize
        if index is not None:
            if index in indices:
                selected_index = index
            else:
                print(f"Index {index} not found in dataset. Selecting a random index.")
                selected_index = random.choice(indices)
        else:
            selected_index = random.choice(indices)
        
        print(f"Visualizing structure with index: {selected_index}")
        
        # Load structure
        structure_path = os.path.join(data_dir, f"data_{selected_index}.pt")
        if not os.path.exists(structure_path):
            print(f"Structure file {structure_path} not found.")
            return
        
        positions, node_features, edge_index, metadata = load_structure(structure_path)
    
    # Set default output path if not specified
    if output_path is None and method != 'py3dmol':
        extension = '.html' if method == 'plotly' else '.png'
        if positions_data is not None and isinstance(positions_data, str):
            basename = os.path.basename(positions_data).split('.')[0]
            output_path = f"structure_{basename}_{method}{extension}"
        else:
            output_path = f"structure_vector_{method}{extension}"
    
    # Visualize structure with the selected method
    if method == 'matplotlib':
        plot_structure_matplotlib(positions, node_features, edge_index, metadata, 
                                output_path, show_edges)
    elif method == 'ase':
        plot_structure_ase(positions, node_features, metadata, output_path, show_edges=show_edges)
    elif method == 'plotly':
        plot_structure_plotly(positions, node_features, edge_index, metadata, 
                            output_path, show_edges)
    elif method == 'py3dmol':
        try:
            plot_structure_py3dmol(positions, node_features, edge_index, metadata)
        except Exception as e:
            print(f"Error using py3Dmol visualization: {e}")
            print("Falling back to matplotlib visualization.")
            plot_structure_matplotlib(positions, node_features, edge_index, metadata, 
                                   output_path, show_edges) 