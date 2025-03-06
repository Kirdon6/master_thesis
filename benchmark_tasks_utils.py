import torch
import torch.nn.functional as F
from torch_geometric.utils import unbatch

def position_MAE(pred_xyz, true_xyz):
    """
    Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
    This function expects flattened tensors of atom coordinates and reshapes them to calculate
    the Manhattan distance (L1 norm) between predicted and true positions for each atom.
    """
    # Reshape the flattened tensors to recover the 3D coordinates
    # Each consecutive triplet of values represents x, y, z coordinates of an atom
    pred_reshaped = pred_xyz.view(-1, 3)
    true_reshaped = true_xyz.view(-1, 3)
    
    # Calculate absolute differences for each coordinate (Manhattan distance)
    abs_diff = torch.abs(pred_reshaped - true_reshaped)
    
    # Sum the absolute differences across coordinates for each atom
    manhattan_distances = torch.sum(abs_diff, dim=1)
    
    # Average these distances to get the MAE in Ångströms
    return torch.mean(manhattan_distances)

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
    
    return pred[truth < 100], truth[truth < 100]

def pos_abs_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from XRD data.
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
    
    return pred[truth < 100], truth[truth < 100]

def pos_abs_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from xPDF data.
    """
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    sct = data.y['xPDF'][:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)

    evaluated_kwargs['x'] = sct
    pred = model(**evaluated_kwargs)
    truth = pos_abs_padded(data, config_dict, device)
    
    return pred[truth < 100], truth[truth < 100] 