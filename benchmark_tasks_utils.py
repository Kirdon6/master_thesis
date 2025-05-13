import torch
from scipy.spatial.distance import directed_hausdorff

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
        
        
        # Return mean across all structures
        return torch.mean(atom_distances)


def hausdorff_distance(
        pred_xyz,
        true_xyz
    ):
        """
        Calculates the Hausdorff distance between the predicted and true positions of the atoms in units of Ångstrøm.
        """

        # Calculate Hausdorff distance for each pair of structures
        hausdorff_distances = []
        for i in range(pred_xyz.shape[0]):
            # Convert to numpy for scipy
            pred_points = pred_xyz[i].cpu().numpy()
            true_points = true_xyz[i].cpu().numpy()
            
            # Calculate directed Hausdorff distance in both directions
            forward_hausdorff = directed_hausdorff(pred_points, true_points)[0]
            backward_hausdorff = directed_hausdorff(true_points, pred_points)[0]
            
            # Take the max of the two directed distances
            hausdorff = max(forward_hausdorff, backward_hausdorff)
            hausdorff_distances.append(hausdorff)
        
        # Calculate mean Hausdorff distance
        mean_hausdorff = sum(hausdorff_distances) / len(hausdorff_distances)
        return mean_hausdorff

def atom_type_accuracy(
        pred_atom_types,
        true_atom_types,
        model_type
    ):
        """
        Calculates the accuracy of the predicted atom types.
        """

        device = pred_atom_types.device
        true_atom_types = true_atom_types.to(device)

        if model_type == 'mlp':
            pred_atom_types = torch.argmax(pred_atom_types, dim=2)  # [batch_size, num_atoms]

        
            
        # Reshape true_types if needed to match pred_classes
        if true_atom_types.dim() == 1 and pred_atom_types.dim() == 2:
            # Expand true_types to match pred_classes shape
            if len(true_atom_types) == pred_atom_types.shape[0]:
                true_atom_types = true_atom_types.unsqueeze(1).expand(-1, pred_atom_types.shape[1])
            elif len(true_atom_types) == pred_atom_types.shape[0] * pred_atom_types.shape[1]:
                true_atom_types = true_atom_types.reshape(pred_atom_types.shape[0], pred_atom_types.shape[1])
        
        # Calculate accuracy
        correct = (pred_atom_types == true_atom_types).float().sum()
        total = true_atom_types.numel()
        
        accuracy = (correct / total) * 100.0
        return accuracy.item() 
        
