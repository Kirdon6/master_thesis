import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
from benchmark_tasks_utils import position_MAE

def convert_image_to_atom_list(self, coord_image: torch.Tensor, type_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert image format (where each pixel is an atom) to atom list format
        
        Args:
            coord_image: (batch, 3, height, width) - coordinates at each pixel
            type_image: (batch, num_types, height, width) - type logits at each pixel
            
        Returns:
            coords: (batch, height*width, 3) - atom coordinates
            types: (batch, height*width, num_types) - atom type logits
        """
        batch_size, _, height, width = coord_image.shape
        num_atoms = height * width
        
        # Reshape coordinates: (batch, 3, h, w) -> (batch, h*w, 3)
        coords = coord_image.permute(0, 2, 3, 1).reshape(batch_size, num_atoms, 3)
        
        # Reshape types: (batch, num_types, h, w) -> (batch, h*w, num_types)
        types = type_image.permute(0, 2, 3, 1).reshape(batch_size, num_atoms, -1)
        
        return coords, types
    
def evaluate_diffusion_format(self, pred_coord_images: torch.Tensor, true_coord_images: torch.Tensor,
                                pred_type_images: torch.Tensor, true_type_images: torch.Tensor,
                                type_mismatch_penalty: float = 20.0) -> Dict:
        """
        Evaluate diffusion model outputs in image format (each pixel is an atom)
        
        Args:
            pred_coord_images: (batch, 3, height, width) predicted coordinates
            true_coord_images: (batch, 3, height, width) true coordinates
            pred_type_images: (batch, num_types, height, width) predicted type logits
            true_type_images: (batch, num_types, height, width) true type logits
            type_mismatch_penalty: penalty for type mismatches
        """
        # Convert to atom list format
        pred_coords, pred_types = self.convert_image_to_atom_list(pred_coord_images, pred_type_images)
        true_coords, true_types = self.convert_image_to_atom_list(true_coord_images, true_type_images)
        
        # Run standard evaluation
        return self.batched_comprehensive_evaluation(
            pred_coords, true_coords, pred_types, true_types, type_mismatch_penalty
        )


class BatchedNanomaterialEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def batched_kabsch_align(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Batch Kabsch alignment for multiple structures
        
        Args:
            P: (batch_size, n_atoms, 3) predicted coordinates
            Q: (batch_size, n_atoms, 3) true coordinates
            
        Returns:
            P_aligned: (batch_size, n_atoms, 3) aligned predicted coordinates
        """
        batch_size = P.shape[0]
        device = P.device
        
        # Calculate centers of mass for each structure in batch
        P_center = P.mean(dim=1, keepdim=True)  # (batch_size, 1, 3)
        Q_center = Q.mean(dim=1, keepdim=True)  # (batch_size, 1, 3)
        
        # Center structures
        P_centered = P - P_center  # (batch_size, n_atoms, 3)
        Q_centered = Q - Q_center  # (batch_size, n_atoms, 3)
        
        # Compute cross-covariance matrices for each pair
        # H[i] = P_centered[i].T @ Q_centered[i]
        H = torch.bmm(P_centered.transpose(1, 2), Q_centered)  # (batch_size, 3, 3)
        
        # SVD for each matrix in the batch
        U, S, Vt = torch.svd(H)
        
        # Ensure proper rotation (no reflection) for each matrix
        det = torch.det(torch.bmm(Vt, U.transpose(1, 2)))  # (batch_size,)
        
        # Flip last column of Vt where determinant is negative
        fix_reflect = det < 0
        Vt_corrected = Vt.clone()
        Vt_corrected[fix_reflect, :, -1] *= -1
        
        # Compute rotation matrices
        R = torch.bmm(Vt_corrected, U.transpose(1, 2))  # (batch_size, 3, 3)
        
        # Apply transformations to get aligned coordinates
        # P_aligned = P_centered @ R.T + Q_center
        P_aligned = torch.bmm(P_centered, R.transpose(1, 2)) + Q_center
        
        return P_aligned
    
    def batched_optimal_assignment(self, P_aligned: torch.Tensor, Q: torch.Tensor) -> Tuple[List, List]:
        """
        Find optimal assignment for each structure in batch
        
        Args:
            P_aligned: (batch_size, n_atoms, 3) aligned predicted coordinates
            Q: (batch_size, n_atoms, 3) true coordinates
            
        Returns:
            pred_indices_list: List of pred indices for each batch
            true_indices_list: List of true indices for each batch
        """
        batch_size = P_aligned.shape[0]
        pred_indices_list = []
        true_indices_list = []
        
        for i in range(batch_size):
            # Calculate distance matrix for this pair
            dist_matrix = torch.cdist(P_aligned[i], Q[i])  # (n_atoms, n_atoms)
            
            # Find optimal assignment using Hungarian algorithm
            pred_indices, true_indices = linear_sum_assignment(dist_matrix.cpu().numpy())
            
            pred_indices_list.append(pred_indices)
            true_indices_list.append(true_indices)
        
        return pred_indices_list, true_indices_list
    
    def type_aware_assignment_with_counts(self, P_aligned: torch.Tensor, Q: torch.Tensor,
                                        pred_types: torch.Tensor, true_types: torch.Tensor) -> Dict:
        """
        Assignment that explicitly handles different atom counts per type
        """
        # Handle case where pred_types might be logits
        if pred_types.dim() > 1:  # If pred_types has shape [n_atoms, n_classes]
            pred_types = torch.argmax(pred_types, dim=-1)  # Convert to class indices
        
        unique_types = torch.unique(torch.cat([pred_types, true_types]))
        
        total_distance = 0
        total_atoms = 0
        type_specific_results = {}
        unmatched_atoms = {'pred': [], 'true': []}
        
        all_pred_indices = []
        all_true_indices = []
        
        for atom_type in unique_types:
            pred_mask = pred_types == atom_type
            true_mask = true_types == atom_type
            
            pred_coords = P_aligned[pred_mask]
            true_coords = Q[true_mask]
            
            pred_count = pred_coords.shape[0]
            true_count = true_coords.shape[0]
            
            if pred_count == 0 and true_count == 0:
                continue
            elif pred_count == 0:
                # Model didn't predict any atoms of this type
                unmatched_atoms['true'].extend(torch.where(true_mask)[0].tolist())
                type_specific_results[atom_type.item()] = {
                    'pred_count': 0,
                    'true_count': true_count,
                    'matched': 0,
                    'avg_distance': float('inf'),
                    'distances': [],  # Add empty distances array
                    'matched_pred_indices': [],
                    'matched_true_indices': []
                }
            elif true_count == 0:
                # Model predicted atoms of a type that shouldn't exist
                unmatched_atoms['pred'].extend(torch.where(pred_mask)[0].tolist())
                type_specific_results[atom_type.item()] = {
                    'pred_count': pred_count,
                    'true_count': 0,
                    'matched': 0,
                    'avg_distance': float('inf'),
                    'distances': [],  # Add empty distances array
                    'matched_pred_indices': [],
                    'matched_true_indices': []
                }
            else:
                # Both have atoms of this type - find optimal matching
                if pred_count <= true_count:
                    # More true atoms than predicted - some true atoms won't be matched
                    dist_matrix = torch.cdist(pred_coords, true_coords)
                    pred_idx, true_idx = linear_sum_assignment(dist_matrix.cpu().numpy())
                    
                    # All predicted atoms are matched
                    matched_pred_indices = torch.where(pred_mask)[0][pred_idx]
                    matched_true_indices = torch.where(true_mask)[0][true_idx]
                    
                    # Record unmatched true atoms
                    unmatched_true_mask = torch.ones(true_count, dtype=bool)
                    unmatched_true_mask[true_idx] = False
                    unmatched_true_atoms = torch.where(true_mask)[0][unmatched_true_mask]
                    unmatched_atoms['true'].extend(unmatched_true_atoms.tolist())
                    
                else:
                    # More predicted atoms than true - some predicted atoms won't be matched
                    dist_matrix = torch.cdist(true_coords, pred_coords).T  # Transpose for row-wise assignment
                    pred_idx, true_idx = linear_sum_assignment(dist_matrix.cpu().numpy())
                    
                    # All true atoms are matched
                    matched_pred_indices = torch.where(pred_mask)[0][pred_idx]
                    matched_true_indices = torch.where(true_mask)[0][true_idx]
                    
                    # Record unmatched pred atoms
                    unmatched_pred_mask = torch.ones(pred_count, dtype=bool)
                    unmatched_pred_mask[pred_idx] = False
                    unmatched_pred_atoms = torch.where(pred_mask)[0][unmatched_pred_mask]
                    unmatched_atoms['pred'].extend(unmatched_pred_atoms.tolist())
                
                # Calculate distances for matched atoms
                distances = torch.norm(pred_coords[pred_idx] - true_coords[true_idx], dim=1)
                avg_distance = distances.mean().item()
                total_distance += distances.sum().item()
                total_atoms += len(distances)
                
                all_pred_indices.extend(matched_pred_indices.tolist())
                all_true_indices.extend(matched_true_indices.tolist())
                
                type_specific_results[atom_type.item()] = {
                    'pred_count': pred_count,
                    'true_count': true_count,
                    'matched': len(pred_idx),
                    'avg_distance': avg_distance,
                    'distances': distances.cpu().numpy(),
                    'matched_pred_indices': matched_pred_indices.tolist(),
                    'matched_true_indices': matched_true_indices.tolist()
                }
        
        return {
            'avg_distance': total_distance / total_atoms if total_atoms > 0 else float('inf'),
            'total_matched_atoms': total_atoms,
            'type_specific': type_specific_results,
            'unmatched_atoms': unmatched_atoms,
            'matched_indices': (all_pred_indices, all_true_indices)
        }
    
    def complete_assignment_with_type_penalties(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor,
                                              pred_types: torch.Tensor, true_types: torch.Tensor,
                                              type_mismatch_penalty: float = 20.0,
                                              align_first: bool = True) -> Dict:
        """
        Two-stage assignment:
        1. Match atoms within same types optimally
        2. Force assignment of remaining unmatched atoms with type penalties
        
        Args:
            pred_xyz: (n_atoms, 3) predicted coordinates
            true_xyz: (n_atoms, 3) true coordinates
            pred_types: (n_atoms,) predicted atom types or (n_atoms, n_classes) logits
            true_types: (n_atoms,) true atom types
            type_mismatch_penalty: Large distance penalty for mismatched types
            align_first: whether to align structures first
        """
        
        # Handle case where pred_types might be logits
        if pred_types.dim() > 1:  # If pred_types has shape [n_atoms, n_classes]
            pred_types = torch.argmax(pred_types, dim=-1)  # Convert to class indices
        
        # Optional alignment first
        if align_first:
            pred_aligned = self.batched_kabsch_align(pred_xyz.unsqueeze(0), true_xyz.unsqueeze(0))[0]
        else:
            pred_aligned = pred_xyz
        
        # Stage 1: Optimal assignment within each type
        type_results = self.type_aware_assignment_with_counts(pred_aligned, true_xyz, pred_types, true_types)
        
        # Track all matched atoms
        matched_pred_indices = set()
        matched_true_indices = set()
        stage1_distances = []
        
        # Collect all same-type matches
        for atom_type, result in type_results['type_specific'].items():
            if 'matched_pred_indices' in result and 'matched_true_indices' in result:
                matched_pred_indices.update(result['matched_pred_indices'])
                matched_true_indices.update(result['matched_true_indices'])
                if 'distances' in result and len(result['distances']) > 0:
                    stage1_distances.extend(result['distances'])
        
        # Stage 2: Handle unmatched atoms
        unmatched_pred = [i for i in range(len(pred_xyz)) if i not in matched_pred_indices]
        unmatched_true = [i for i in range(len(true_xyz)) if i not in matched_true_indices]
        
        stage2_distances = []
        stage2_type_matches = 0
        stage2_assignments = ([], [])
        
        if unmatched_pred and unmatched_true:
            # Create distance matrix for unmatched atoms
            unmatched_pred_coords = pred_aligned[unmatched_pred]
            unmatched_true_coords = true_xyz[unmatched_true]
            
            # Calculate spatial distances
            spatial_distances = torch.cdist(unmatched_pred_coords, unmatched_true_coords)
            
            # Calculate type mismatch penalties
            unmatched_pred_types = pred_types[unmatched_pred]
            unmatched_true_types = true_types[unmatched_true]
            
            type_mismatches = (unmatched_pred_types.unsqueeze(1) != unmatched_true_types.unsqueeze(0)).float()
            type_penalties = type_mismatches * type_mismatch_penalty
            
            # Combined cost matrix
            cost_matrix = spatial_distances + type_penalties
            
            # Find optimal assignment for unmatched atoms
            pred_idx, true_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # Calculate distances and track type matches
            for pi, ti in zip(pred_idx, true_idx):
                spatial_dist = spatial_distances[pi, ti].item()
                type_match = unmatched_pred_types[pi] == unmatched_true_types[ti]
                
                if type_match:
                    stage2_distances.append(spatial_dist)
                    stage2_type_matches += 1
                else:
                    # Include penalty in the distance for mismatched types
                    stage2_distances.append(spatial_dist + type_mismatch_penalty)
            
            # Record assignments (convert back to global indices)
            stage2_assignments = (
                [unmatched_pred[i] for i in pred_idx],
                [unmatched_true[i] for i in true_idx]
            )
        
        # Combine results
        all_distances = stage1_distances + stage2_distances
        total_same_type_matches = sum(result.get('matched', 0) for result in type_results['type_specific'].values())
        total_different_type_matches = len(stage2_distances) - stage2_type_matches
        
        return {
            'mean_distance': np.mean(all_distances) if all_distances else float('inf'),
            'total_atoms': len(pred_xyz),
            'matched_atoms': len(all_distances),
            'unmatched_pred': len(unmatched_pred) - len(stage2_assignments[0]),
            'unmatched_true': len(unmatched_true) - len(stage2_assignments[1]),
            
            # Type-specific metrics
            'same_type_matches': total_same_type_matches,
            'different_type_matches': total_different_type_matches,
            'type_accuracy': total_same_type_matches / len(all_distances) if all_distances else 0.0,
            
            # Detailed breakdown
            'stage1_distances': stage1_distances,
            'stage2_distances': stage2_distances,
            'stage1_avg': np.mean(stage1_distances) if stage1_distances else 0.0,
            'stage2_avg': np.mean(stage2_distances) if stage2_distances else 0.0,
            
            # Assignment information
            'stage1_assignments': type_results.get('matched_indices', ([], [])),
            'stage2_assignments': stage2_assignments,
            
            # Type-specific breakdown
            'type_specific': type_results['type_specific']
        }
    
    def batched_complete_assignment_with_type_penalties(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor,
                                                      pred_types: torch.Tensor, true_types: torch.Tensor,
                                                      type_mismatch_penalty: float = 20.0,
                                                      align_first: bool = True) -> Dict:
        """
        Batch version of complete assignment with type penalties
        
        Args:
            pred_xyz: (batch_size, n_atoms, 3) predicted coordinates
            true_xyz: (batch_size, n_atoms, 3) true coordinates
            pred_types: (batch_size, n_atoms) or (batch_size, n_atoms, n_classes) predicted atom types/logits
            true_types: (batch_size, n_atoms) true atom types
            type_mismatch_penalty: Large distance penalty for mismatched types
            align_first: whether to align structures first
        """
        batch_size = pred_xyz.shape[0]
        batch_results = []
        
        # Handle case where pred_types might be logits
        if pred_types.dim() > 2:  # If pred_types has shape [batch_size, n_atoms, n_classes]
            pred_types = torch.argmax(pred_types, dim=-1)  # Convert to class indices
        
        for i in range(batch_size):
            result = self.complete_assignment_with_type_penalties(
                pred_xyz[i], true_xyz[i], pred_types[i], true_types[i], 
                type_mismatch_penalty, align_first
            )
            batch_results.append(result)
        
        # Aggregate batch statistics
        return {
            'batch_results': batch_results,
            'batch_mean_distance': np.mean([r['mean_distance'] for r in batch_results]),
            'batch_type_accuracy': np.mean([r['type_accuracy'] for r in batch_results]),
            'batch_same_type_ratio': np.mean([r['same_type_matches'] / r['matched_atoms'] 
                                             for r in batch_results if r['matched_atoms'] > 0]),
            'batch_unmatched_ratio': np.mean([(r['unmatched_pred'] + r['unmatched_true']) / (r['total_atoms'] * 2) 
                                             for r in batch_results]),
            'batch_std_distance': np.std([r['mean_distance'] for r in batch_results]),
            'batch_min_distance': np.min([r['mean_distance'] for r in batch_results]),
            'batch_max_distance': np.max([r['mean_distance'] for r in batch_results])
        }
    
    def batched_optimal_assignment_distance(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor, 
                                          align_first: bool = True) -> Dict:
        """
        Calculate optimal assignment distances for batch of structures
        
        Args:
            pred_xyz: (batch_size, n_atoms, 3) predicted coordinates
            true_xyz: (batch_size, n_atoms, 3) true coordinates
            align_first: whether to align structures first
            
        Returns:
            Dictionary with batch results
        """
        batch_size = pred_xyz.shape[0]
        
        # Step 1: Optionally align structures
        if align_first:
            pred_aligned = self.batched_kabsch_align(pred_xyz, true_xyz)
        else:
            pred_aligned = pred_xyz
        
        # Step 2: Find optimal assignments for each structure
        pred_indices_list, true_indices_list = self.batched_optimal_assignment(pred_aligned, true_xyz)
        
        # Step 3: Calculate distances for each structure
        all_distances = []
        mean_distances = []
        max_distances = []
        min_distances = []
        std_distances = []
        
        for i in range(batch_size):
            # Extract assigned coordinates
            pred_assigned = pred_aligned[i][pred_indices_list[i]]
            true_assigned = true_xyz[i][true_indices_list[i]]
            
            # Calculate distances
            distances = torch.norm(pred_assigned - true_assigned, dim=1)
            all_distances.append(distances.cpu().numpy())
            
            mean_distances.append(distances.mean().item())
            max_distances.append(distances.max().item())
            min_distances.append(distances.min().item())
            std_distances.append(distances.std().item())
        
        return {
            'mean_distances': mean_distances,  # List of means for each structure
            'max_distances': max_distances,
            'min_distances': min_distances,
            'std_distances': std_distances,
            'assignments': (pred_indices_list, true_indices_list),
            'all_distances': all_distances,
            # Aggregated statistics
            'batch_mean': np.mean(mean_distances),
            'batch_std': np.std(mean_distances),
            'batch_min': np.min(min_distances),
            'batch_max': np.max(max_distances)
        }
    
    def batched_comprehensive_evaluation(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor,
                                       pred_types: Optional[torch.Tensor] = None,
                                       true_types: Optional[torch.Tensor] = None,
                                       type_mismatch_penalty: float = 20.0) -> Dict:
        """
        Comprehensive evaluation for batch of structures
        
        Args:
            pred_xyz: (batch_size, n_atoms, 3) predicted coordinates
            true_xyz: (batch_size, n_atoms, 3) true coordinates
            pred_types: (batch_size, n_atoms) predicted atom types (optional)
            true_types: (batch_size, n_atoms) true atom types (optional)
            type_mismatch_penalty: penalty for type mismatches when using types
            
        Returns:
            Dictionary with comprehensive metrics
        """
        batch_size = pred_xyz.shape[0]
        results = {}
        
        # 1. Alignment + Assignment distance (without types)
        aligned_results = self.batched_optimal_assignment_distance(pred_xyz, true_xyz, align_first=True)
        results['aligned_assignment'] = aligned_results
        
        # 2. Assignment without alignment (without types)
        unaligned_results = self.batched_optimal_assignment_distance(pred_xyz, true_xyz, align_first=False)
        results['unaligned_assignment'] = unaligned_results

        # 3. Position MAE
        position_mae = position_MAE(pred_xyz, true_xyz)
        results['position_mae'] = {
            'values': position_mae.cpu().numpy().tolist(),
            'batch_mean': position_mae.mean().item(),
        }
        
        # 4. Naive RMSD (assuming correct order)
        naive_rmsd = torch.sqrt(torch.mean(torch.sum((pred_xyz - true_xyz)**2, dim=2), dim=1))
        results['naive_rmsd'] = {
            'values': naive_rmsd.cpu().numpy().tolist(),
            'batch_mean': naive_rmsd.mean().item(),
            'batch_std': naive_rmsd.std().item()
        }
        
        # 5. Center of mass differences
        pred_com = pred_xyz.mean(dim=1)  # (batch_size, 3)
        true_com = true_xyz.mean(dim=1)  # (batch_size, 3)
        com_diffs = torch.norm(pred_com - true_com, dim=1)
        results['center_of_mass_diff'] = {
            'values': com_diffs.cpu().numpy().tolist(),
            'batch_mean': com_diffs.mean().item(),
            'batch_std': com_diffs.std().item()
        }
        
        # 6. Radius of gyration differences
        def batched_radius_of_gyration(coords):
            # coords: (batch_size, n_atoms, 3)
            centered = coords - coords.mean(dim=1, keepdim=True)
            return torch.sqrt(torch.mean(torch.sum(centered**2, dim=2), dim=1))
        
        pred_rog = batched_radius_of_gyration(pred_xyz)
        true_rog = batched_radius_of_gyration(true_xyz)
        rog_diffs = torch.abs(pred_rog - true_rog)
        results['radius_of_gyration_diff'] = {
            'values': rog_diffs.cpu().numpy().tolist(),
            'batch_mean': rog_diffs.mean().item(),
            'batch_std': rog_diffs.std().item()
        }
        
        # 7. Type-aware metrics (if types are provided)
        if pred_types is not None and true_types is not None:
            type_aware_results = self.batched_complete_assignment_with_type_penalties(
                pred_xyz, true_xyz, pred_types, true_types, type_mismatch_penalty, align_first=True
            )
            results['type_aware_assignment'] = type_aware_results
            
            # Type-only accuracy (perfect geometric matching)
            type_only_accuracy = (pred_types == true_types).float().mean(dim=1)
            results['type_only_accuracy'] = {
                'values': type_only_accuracy.cpu().numpy().tolist(),
                'batch_mean': type_only_accuracy.mean().item(),
                'batch_std': type_only_accuracy.std().item()
            }
        
        return results

def quick_batch_metric(pred_batch: torch.Tensor, true_batch: torch.Tensor,
                     input_format: str = 'auto') -> float:
    """
    Get single metric for batch evaluation (mean aligned distance)
    """
    evaluator = BatchedNanomaterialEvaluator()
    
    # Handle different formats
    if input_format == 'auto':
        input_format = 'images' if pred_batch.dim() == 4 else 'atoms'
    
    if input_format == 'images':
        # Convert to atom format
        pred_coords, _ = evaluator.convert_image_to_atom_list(pred_batch, pred_batch)  # Use coords as dummy types
        true_coords, _ = evaluator.convert_image_to_atom_list(true_batch, true_batch)
        results = evaluator.batched_optimal_assignment_distance(pred_coords, true_coords)
    else:
        results = evaluator.batched_optimal_assignment_distance(pred_batch, true_batch)
    
    return results['batch_mean']
def evaluate_diffusion_batch(pred_batch: torch.Tensor, true_batch: torch.Tensor,
                           pred_types: Optional[torch.Tensor] = None,
                           true_types: Optional[torch.Tensor] = None,
                           input_format: str = 'auto') -> Dict:
    """
    Quick evaluation for a batch of generated structures
    
    Args:
        pred_batch: predicted data
        true_batch: true data  
        pred_types: predicted atom types (optional)
        true_types: true atom types (optional)
        input_format: 'atoms', 'images', or 'auto' to detect automatically
    """
    evaluator = BatchedNanomaterialEvaluator()
    
    # Auto-detect format if not specified
    if input_format == 'auto':
        # Check dimensions to determine format
        if pred_batch.dim() == 4:  # (batch, channels, height, width)
            input_format = 'images'
        elif pred_batch.dim() == 3:  # (batch, atoms, features)
            input_format = 'atoms'
        else:
            raise ValueError(f"Cannot auto-detect format for tensor with {pred_batch.dim()} dimensions")
    
    if input_format == 'images':
        return evaluator.evaluate_diffusion_format(pred_batch, true_batch, pred_types, true_types)
    else:
        return evaluator.batched_comprehensive_evaluation(pred_batch, true_batch, pred_types, true_types)

def quick_batch_metric_with_types(pred_batch: torch.Tensor, true_batch: torch.Tensor,
                                pred_types: torch.Tensor, true_types: torch.Tensor,
                                input_format: str = 'auto') -> Dict:
    """
    Get quick metrics with type information for different input formats
    """
    evaluator = BatchedNanomaterialEvaluator()
    
    # Auto-detect format if not specified
    if input_format == 'auto':
        if pred_batch.dim() == 4:  # (batch, channels, height, width)
            input_format = 'images'
        elif pred_batch.dim() == 3:  # (batch, atoms, features)
            input_format = 'atoms'
        else:
            raise ValueError(f"Cannot auto-detect format for tensor with {pred_batch.dim()} dimensions")
    
    if input_format == 'images':
        # Convert image format to atom list format
        pred_coords, pred_types_converted = evaluator.convert_image_to_atom_list(pred_batch, pred_types)
        true_coords, true_types_converted = evaluator.convert_image_to_atom_list(true_batch, true_types)
        
        # Use the converted data
        results = evaluator.batched_complete_assignment_with_type_penalties(
            pred_coords, true_coords, pred_types_converted, true_types_converted
        )
    else:
        # Use data as-is for atom list format
        results = evaluator.batched_complete_assignment_with_type_penalties(
            pred_batch, true_batch, pred_types, true_types
        )
    
    return {
        'mean_distance': results['batch_mean_distance'],
        'type_accuracy': results['batch_type_accuracy']
    }

# Example usage:
if __name__ == "__main__":
    # Example with batch of structures
    batch_size = 16
    n_atoms = 50
    n_types = 5  # e.g., C, O, N, H, S
    
    # Simulate batch data
    pred_batch = torch.randn(batch_size, n_atoms, 3, device='cuda')
    true_batch = torch.randn(batch_size, n_atoms, 3, device='cuda')
    pred_types = torch.randint(0, n_types, (batch_size, n_atoms), device='cuda')
    true_types = torch.randint(0, n_types, (batch_size, n_atoms), device='cuda')
    
    # Full evaluation
    evaluator = BatchedNanomaterialEvaluator()
    results = evaluator.batched_comprehensive_evaluation(pred_batch, true_batch, pred_types, true_types)
    
    print("Batch Evaluation Results:")
    print(f"Mean aligned distance across batch: {results['aligned_assignment']['batch_mean']:.4f}")
    print(f"Type-aware aligned distance: {results['type_aware_assignment']['batch_mean_distance']:.4f}")
    print(f"Type accuracy: {results['type_aware_assignment']['batch_type_accuracy']:.4f}")
    print(f"Position MAE: {results['position_mae']['batch_mean']:.4f}")
    
    # Quick single metric
    quick_metric = quick_batch_metric(pred_batch, true_batch)
    print(f"Quick batch metric: {quick_metric:.4f}")
    
    # Quick metric with types
    quick_type_metric = quick_batch_metric_with_types(pred_batch, true_batch, pred_types, true_types)
    print(f"Quick batch metric with types: {quick_type_metric['mean_distance']:.4f}")
    print(f"Type accuracy: {quick_type_metric['type_accuracy']:.4f}")