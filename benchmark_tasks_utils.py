import torch
import torch.nn.functional as F
from torch_geometric.utils import unbatch

def position_MAE(
        pred_xyz,
        true_xyz
    ):
        """
        Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
        Ignores padded atoms (where coordinates in true_xyz are 0) and calculates MAE per structure.
        Expects inputs in format [batch_size, atoms*3].
        """
        # Reshape from [batch, atoms*3] to [batch, atoms, 3]
        batch_size = pred_xyz.shape[0]
        coords_per_atom = 3
        num_atoms = pred_xyz.shape[1] // coords_per_atom
        
        pred_xyz = pred_xyz.view(batch_size, num_atoms, coords_per_atom)
        true_xyz = true_xyz.view(batch_size, num_atoms, coords_per_atom)
        
        # Create mask for non-padded atoms (where coordinates are not 100)
        valid_mask = (true_xyz < 100).all(dim=2)
        
        # Calculate Euclidean distances for all atoms
        atom_distances = torch.sqrt(torch.sum((pred_xyz - true_xyz)**2, dim=2))
        
        # Apply mask to get only valid atoms
        # Set distances for padded atoms to 0
        masked_distances = atom_distances * valid_mask.float()
        
        # Count valid atoms per structure
        valid_atoms_per_structure = valid_mask.sum(dim=1).float()
        
        # Sum distances per structure and divide by number of valid atoms
        # This gives us the MAE per structure
        structure_maes = masked_distances.sum(dim=1) / valid_atoms_per_structure
        
        # Return mean across all structures
        return torch.mean(structure_maes)

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
    
    # Get xPDF data
    xpdf = data.y['xPDF']
    
    # Normalize xPDF data
    sct = xpdf[:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)
    
    # For VectorDiffusion, we need to pass the full xPDF data
    if config_dict["model"] == "VectorDiffusion":

        
        # Use the normalized xPDF for the model
        evaluated_kwargs['x'] = sct       

        # For inference, use the forward method
        pred = model(**evaluated_kwargs)
        truth = pos_abs_padded(data, config_dict, device)
        return pred, truth
    else:
        # For other models, use the standard approach
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = pos_abs_padded(data, config_dict, device)
        return pred, truth 