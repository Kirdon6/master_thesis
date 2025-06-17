import json
from tqdm import tqdm
from CHILI_centralAtoms import CHILI
import yaml
from torch_geometric.loader import DataLoader

def create_atom_type_mapping(data_loader, output_path="atom_type_mapping.json"):
    """
    Creates a mapping between atomic numbers and integer categories from CHILI3K dataset.
    
    Parameters:
    -----------
    data_loader : DataLoader or list
        Data loader containing CHILI3K dataset batches
    output_path : str
        Path to save the mapping JSON file
    
    Returns:
    --------
    dict
        Dictionary with mappings: 
        - atom_num_to_idx: Maps atomic numbers to integer indices
        - idx_to_atom_num: Maps integer indices to atomic numbers
        - num_categories: Total number of atom types
    """
    print("Analyzing dataset to find all unique atomic numbers...")
    all_atomic_numbers = set()
    
    # Go through all batches to find unique atomic numbers
    for batch in tqdm(data_loader):
        # Extract atomic numbers from x, assuming x has shape [num_atoms, 4]
        # where the last dimension contains [x, y, z, atomic_number]
        if hasattr(batch, 'x'):
            # Extract all atomic numbers from this batch
            atomic_numbers = batch.x[:, 3].cpu().numpy()
            
            # Add to set of all atomic numbers
            for atom_num in atomic_numbers:
                all_atomic_numbers.add(int(atom_num))
    
    # Sort atomic numbers
    all_atomic_numbers = sorted(list(all_atomic_numbers))
    
    # Create mappings
    atom_num_to_idx = {int(atom_num): idx for idx, atom_num in enumerate(all_atomic_numbers)}
    idx_to_atom_num = {idx: int(atom_num) for idx, atom_num in enumerate(all_atomic_numbers)}
    
    # Create the final mapping dictionary
    mapping = {
        "atom_num_to_idx": atom_num_to_idx,
        "idx_to_atom_num": idx_to_atom_num,
        "num_categories": len(all_atomic_numbers)
    }
    
    # Save the mapping to a JSON file
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Found {len(all_atomic_numbers)} unique atomic numbers.")
        print(f"Mapping saved to {output_path}")
    
    return mapping

def create_mapping_from_all_splits(train_loader, val_loader, test_loader, output_path="atom_type_mapping.json"):
    """
    Creates a unified atom type mapping from all dataset splits (train, validation, test)
    
    Parameters:
    -----------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    output_path : str
        Path to save the mapping JSON file
        
    Returns:
    --------
    dict
        The mapping dictionary
    """
    # Collect all atomic numbers from all splits
    all_atomic_numbers = set()
    
    # Process each loader and collect atomic numbers
    for split_name, loader in [("Training", train_loader), 
                              ("Validation", val_loader), 
                              ("Test", test_loader)]:
        print(f"Processing {split_name} set...")
        for batch in tqdm(loader):
            if hasattr(batch, 'x'):
                atomic_numbers = batch.x[:, 0].cpu().numpy()
                for atom_num in atomic_numbers:
                    all_atomic_numbers.add(int(atom_num))
    
    # Sort atomic numbers
    all_atomic_numbers = sorted(list(all_atomic_numbers))
    
    # Create mappings
    atom_num_to_idx = {int(atom_num): idx for idx, atom_num in enumerate(all_atomic_numbers)}
    idx_to_atom_num = {idx: int(atom_num) for idx, atom_num in enumerate(all_atomic_numbers)}
    
    # Create the final mapping dictionary
    mapping = {
        "atom_num_to_idx": atom_num_to_idx,
        "idx_to_atom_num": idx_to_atom_num,
        "num_categories": len(all_atomic_numbers)
    }
    
    # Save the mapping to a JSON file
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Found {len(all_atomic_numbers)} unique atomic numbers across all dataset splits.")
    print(f"Unified mapping saved to {output_path}")
    
    return mapping

def create_mapping_from_config(config_path, output_path="atom_type_mapping.json"):
    """
    Creates atom type mapping using a configuration file, considering all dataset splits
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
    output_path : str
        Path to save the mapping JSON file
        
    Returns:
    --------
    dict
        The mapping dictionary
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create dataset
    dataset = CHILI(root=config["root"], dataset=config["dataset"], graph_type=config["graph_type"])
    
    # Try to load data split, or handle if not already split
    try:
        dataset.load_data_split(split_strategy='random')
        print("Successfully loaded existing data split.")
    except FileNotFoundError:
        print("No data split found. You may need to run dataset.create_data_split() first.")
        # Return a mapping just from the full dataset
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        return create_atom_type_mapping(loader, output_path)
    
    # Create loaders for each split
    batch_size = config.get("Train_config", {}).get("batch_size", 32)
    train_loader = DataLoader(dataset.train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset.validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=batch_size, shuffle=False)
    
    # Create and return unified mapping
    return create_mapping_from_all_splits(
        train_loader, val_loader, test_loader, output_path
    )
