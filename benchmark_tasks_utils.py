import torch
import torch.nn.functional as F
from torch_geometric.utils import unbatch

# def position_MAE(pred_xyz, true_xyz):
#     """
#     Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
#     This function expects flattened tensors of atom coordinates and reshapes them to calculate
#     the Euclidean distance between predicted and true positions for each atom.
#     """
#     # Reshape the flattened tensors to recover the 3D coordinates
#     # Each consecutive triplet of values represents x, y, z coordinates of an atom
#     pred_reshaped = pred_xyz.view(-1, 3)
#     true_reshaped = true_xyz.view(-1, 3)
    
#     # Calculate squared differences for each coordinate
#     squared_diff = (pred_reshaped - true_reshaped) ** 2
    
#     # Sum across the coordinate dimensions (x, y, z) for each atom
#     summed_squared_diff = torch.sum(squared_diff, dim=1)
    
#     # Take the square root to get the Euclidean distance for each atom
#     distances = torch.sqrt(summed_squared_diff)
    
#     # Average these distances to get the MAE in Ångströms
#     return torch.mean(distances)


# def position_MAE(
#     pred_xyz,
#     true_xyz
# ):
#     """
#     Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
#     """
#     return torch.mean(
#         torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction="none"), dim=1)),
#         dim=0,
#     )

def position_MAE(
        pred_xyz,
        true_xyz
    ):
        """
        Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
        Ignores padded atoms (where coordinates in true_xyz are 0) and calculates MAE per structure.
        Expects inputs in format [batch_size, atoms*3].
        """
        # print(f"Input shapes: {pred_xyz.shape}, {true_xyz.shape}")
        
        # Reshape from [batch, atoms*3] to [batch, atoms, 3]
        batch_size = pred_xyz.shape[0]
        coords_per_atom = 3
        num_atoms = pred_xyz.shape[1] // coords_per_atom
        
        pred_xyz = pred_xyz.view(batch_size, num_atoms, coords_per_atom)
        true_xyz = true_xyz.view(batch_size, num_atoms, coords_per_atom)
        # print(f"pred_xyz shape: {pred_xyz.shape}, true_xyz shape: {true_xyz.shape}")
        
        # Create mask for non-padded atoms (where coordinates are not 100)
        valid_mask = (true_xyz < 100).all(dim=2)
        
        # Initialize list to store per-structure MAE
        structure_maes = []
        
        # Calculate MAE for each structure separately
        for i in range(batch_size):
            # Get valid atoms for this structure
            structure_valid_mask = valid_mask[i]
            
            # Skip if no valid atoms (shouldn't happen in practice)
            if not torch.any(structure_valid_mask):
                continue
                
            # Get predictions and ground truth for valid atoms only
            structure_pred = pred_xyz[i, structure_valid_mask]
            structure_true = true_xyz[i, structure_valid_mask]
            # print(f"structure_pred shape: {structure_pred.shape}, structure_true shape: {structure_true.shape}")
            
            # Calculate Euclidean distance for each atom
            atom_distances = torch.sqrt(torch.sum((structure_pred - structure_true)**2, dim=1))
            
            # Calculate mean for this structure
            structure_mae = torch.mean(atom_distances)
            structure_maes.append(structure_mae)
        
        # Convert list to tensor and calculate mean across all structures

        return torch.mean(torch.stack(structure_maes))

def pos_abs_padded(data, config_dict, device):
    """
    Pads the absolute positions of atoms to a fixed size.
    """
    batch_size = torch.max(data.batch) + 1
    truth = torch.zeros((batch_size, config_dict['Model_config']['out_channels'])).to(device=device)
    for i, x in enumerate(unbatch(data.pos_abs, data.batch)):
        # Sort according to norm
        norms = torch.norm(x, p=2, dim=-1)
        indices = torch.sort(norms, descending=False, dim=0)[1]
        x = x[indices]

        # Padding
        padding_size = config_dict['Model_config']['out_channels'] // 3 - x.size(0)
        if padding_size > 0:
            padding = torch.full((padding_size, x.size(1)), 100, dtype=x.dtype).to(device=device)
            x = torch.cat([x, padding], dim=0)

        # Append
        truth[i] = x.flatten()

    return truth

def pos_abs_from_saxs(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from SAXS data.
    Returns tensors in [batch, atoms*3] format for MAE calculation.
    """
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    sct = data.y['saxs'][:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)

    evaluated_kwargs['x'] = sct
    pred = model(**evaluated_kwargs)
    truth = pos_abs_padded(data, config_dict, device)

    
    return pred, truth

def pos_abs_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from XRD data.
    Returns tensors in [batch, atoms*3] format for MAE calculation.
    """
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    sct = data.y['xrd'][:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)

    evaluated_kwargs['x'] = sct
    pred = model(**evaluated_kwargs)
    truth = pos_abs_padded(data, config_dict, device)
    
    
    
    return pred, truth

def pos_abs_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from xPDF data.
    Returns tensors in [batch, atoms*3] format for MAE calculation.
    """
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    sct = data.y['xPDF'][:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)
    # print(f"input shape: {sct.shape}")

    evaluated_kwargs['x'] = sct
    pred = model(**evaluated_kwargs)
    # print(f"pred shape: {pred.shape}")
    truth = pos_abs_padded(data, config_dict, device)
    # print(f"truth shape: {truth.shape}" )

    return pred, truth 