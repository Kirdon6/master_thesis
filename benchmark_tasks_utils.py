import torch
import torch.nn.functional as F
from torch_geometric.utils import unbatch

def position_MAE(
        pred_xyz,
        true_xyz
    ):
        """
        Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
        Expects inputs in format [batch_size, num_atoms, 3] where all structures have exactly 100 atoms.
        """

        
        # Calculate Euclidean distances for all atoms
        atom_distances = torch.sqrt(torch.sum((pred_xyz - true_xyz)**2, dim=2))
        
        # Average distances for each structure
        #structure_maes = atom_distances.mean(dim=1)
        
        # Return mean across all structures
        return torch.mean(atom_distances)

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
    Returns tensors in [batch_size, num_atoms, 3] format for MAE calculation.
    """
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    
    # Get SAXS data and normalize
    sct = data.y['saxs'][:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)

    # Get ground truth positions
    truth = data.pos_abs
    batch_size = torch.max(data.batch) + 1
    num_atoms = config_dict['Model_config']['out_channels'] // 3
    truth = truth.reshape(batch_size, num_atoms, 3)

    # Pass normalized SAXS to the model
    evaluated_kwargs['x'] = sct
    pred = model(**evaluated_kwargs)


    
    return pred, truth

def pos_abs_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from XRD data.
    Returns tensors in [batch_size, num_atoms, 3] format for MAE calculation.
    """
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    
    # Get XRD data and normalize
    sct = data.y['xrd'][:,1,:]
    sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
    sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
    sct = (sct - sct_min) / (sct_max - sct_min)

    # Get ground truth positions
    truth = data.pos_abs
    batch_size = torch.max(data.batch) + 1
    num_atoms = config_dict['Model_config']['out_channels'] // 3
    truth = truth.reshape(batch_size, num_atoms, 3)

    # Pass normalized XRD to the model
    evaluated_kwargs['x'] = sct
    pred = model(**evaluated_kwargs)

    
    return pred, truth

def pos_abs_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
    """
    Predicts absolute positions from xPDF data.
    Returns tensors in [batch_size, num_atoms, 3] format for MAE calculation.
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
    
    # Get ground truth positions
    truth = data.pos_abs
    batch_size = torch.max(data.batch) + 1
    num_atoms = config_dict['Model_config']['out_channels'] // 3
    truth = truth.reshape(batch_size, num_atoms, 3)
    
    # For VectorDiffusion, we need to handle differently based on train/eval mode
    if config_dict["model"] == "VectorDiffusion":
        if model.training:
            # During training, use the forward_training method which builds a computational graph
            pred = model.forward_training(sct)
        else:
            # During evaluation, use the sampling method
            model.eval()
            pred = model(sct)
    else:
        # For other models, use the standard approach
        pred = model(**evaluated_kwargs)
    
    return pred, truth 