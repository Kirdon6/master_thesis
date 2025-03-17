### Modified CHILI dataset class
### Paper: https://doi.org/10.1145/3637528.3671538
### Data: https://doi.org/10.17894/ucph.e37b6615-8635-49cf-819d-eae60e781a96
### Code: https://github.com/UlrikFriisJensen/CHILI/blob/main/benchmark/dataset_class.py

from typing import Optional, Callable, List, Union
import os
import h5py
import numpy as np
import pandas as pd
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.utils import subgraph, dense_to_sparse
from torch_geometric.data import Data, Dataset, download_url, extract_zip
from tqdm.auto import tqdm
from ase import Atoms

class CHILI(Dataset):
    """
    Dataset class for CHILI dataset.
    """
    def __init__(
        self,
        root: str,
        dataset: str,
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        graph_type: str = "",
    ) -> None:
        """
        Initializes CHILI dataset.
        
        Args:
            root (str): Root directory of the dataset.
            dataset (str): Name of the dataset.
            transform (callable, optional): A function/transform to apply to the data.
            pre_transform (callable, optional): A function/transform to apply to the data before saving.
            pre_filter (callable, optional): A function that takes data and returns True if the data point should be included in the dataset.
            graph_type (str, optional): Type of graph. Defaults to "".
        """
        # Create dataset folder if it does not exists
        if not os.path.exists(root):
            os.mkdir(root)
        
        self.dataset = dataset
        self.root = os.path.join(root, self.dataset)
        # Create root directory if not exits
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        # Train Val Test sets as Subsets
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        # Something is wrong with super, setup manually:
        self.transform = lambda data: data
        self.pre_transform = lambda data: data
        self.pre_filter = lambda data: data

        # Download if data if there are no raw files
        if len(self.raw_file_names) == 0:
            # Make raw folder
            if not os.path.exists(os.path.join(self.root, "raw")):
                os.mkdir(os.path.join(self.root, "raw"))
            # Download
            self.download()
        # Process if processed folder is empty
        if len(self.processed_file_names) == 0:
            # Make processed folder
            if not os.path.exists(os.path.join(self.root, "processed")):
                os.mkdir(os.path.join(self.root, "processed"))
            if not os.path.exists(os.path.join(self.root, "processed_unit_cell")):
                os.mkdir(os.path.join(self.root, "processed_unit_cell"))
            if not os.path.exists(os.path.join(self.root, "processed_central")):
                os.mkdir(os.path.join(self.root, "processed_central"))
            self.process()

        self._indices = range(self.len())
        
        if graph_type.split('-')[0] not in ["", "unit_cell", "central"]:
            raise ValueError(
                'Graph type not recognized. Please use either "", "unit_cell", "central"'
            )
        self.graph_type = graph_type.split('-')[0]

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the list of raw file names in the dataset.
        """
        paths = glob(os.path.join(self.raw_dir, "**/*.h5"))
        return paths

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the list of processed file names in the dataset.
        """
        paths = glob(os.path.join(self.processed_dir, "[!pre]*.pt"))
        return paths

    def download(self) -> None:
        """
        Downloads the dataset.
        """
        # Download to `self.raw_dir`.
        path = download_url(
            f"https://erda.ku.dk/archives/c9d91863f89c3a7e87201c175ff4b213/Nanostructure_Data/Data_for_MachineLearning/DatasetPaper/{self.dataset}.zip",
            self.raw_dir,
        )
        # Extract zip and delete zip
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def crystal_system_to_number(
        self,
        crystal_system: str
    ) -> int:
        """
        Converts crystal system to a number.

        Args:
            crystal_system (str): Crystal system name.

        Returns:
            int: Corresponding crystal system number.
        
        Raises:
            ValueError: If crystal system is not recognized.
        """
        if crystal_system == "Triclinic":
            return 1
        elif crystal_system == "Monoclinic":
            return 2
        elif crystal_system == "Orthorhombic":
            return 3
        elif crystal_system == "Tetragonal":
            return 4
        elif crystal_system == "Trigonal":
            return 5
        elif crystal_system == "Hexagonal":
            return 6
        elif crystal_system == "Cubic":
            return 7
        else:
            raise ValueError(
                'Crystal system not recognized. Please use either "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal" or "Cubic"'
            )

    def write_to_log(
        self,
        log_file: str,
        s: str
    ) -> None:
        """
        Writes the message to a log file.

        Args:
            log_file (str): Path to the log file.
            s (str): Message to be written to the log file.
        """
        try:
            with open(log_file, 'a') as f:
                f.write(s + '\n')
        except Exception as e:
            print(f'Error while writing {s} to file: {log_file}')

    def process(
        self
    ) -> None:
        """
        Processes the raw data and saves the processed data.
        """
        # Find largest number of unit cell atoms in the dataset
        max_unit_cell_n_atoms = 0
        for raw_path in self.raw_file_names:
            _unit_cell_node_feat = torch.tensor(h5py.File(raw_path, "r")["UnitCellGraph"]["NodeFeatures"][:], dtype=torch.float32)
            max_unit_cell_n_atoms = max(max_unit_cell_n_atoms, _unit_cell_node_feat.shape[0])
        
        idx = 0
        process_pbar = tqdm(desc="Processing data...", total=len(self.raw_file_names), leave=False)
        for raw_path in self.raw_file_names:

            # Read data from `raw_path`
            try:
                with h5py.File(raw_path, "r") as h5f:

                    # Unit cell
                    unit_cell_node_feat = torch.tensor(h5f["UnitCellGraph"]["NodeFeatures"][:], dtype=torch.float32)
                    unit_cell_edge_index = torch.tensor(h5f["UnitCellGraph"]["EdgeDirections"][:], dtype=torch.long)
                    unit_cell_edge_attr = torch.tensor(h5f["UnitCellGraph"]["EdgeFeatures"][:], dtype=torch.float32)
                    unit_cell_pos_abs = torch.tensor(h5f["UnitCellGraph"]["AbsoluteCoordinates"][:], dtype=torch.float32)
                    unit_cell_pos_frac = torch.tensor(h5f["UnitCellGraph"]["FractionalCoordinates"][:], dtype=torch.float32)

                    # Cell parameters
                    cell_params = torch.tensor(h5f["GlobalLabels"]["CellParameters"][:], dtype=torch.float32)
                    # Atomic species
                    atomic_species = torch.tensor(h5f["GlobalLabels"]["ElementsPresent"][:], dtype=torch.float32)

                    # Crystal type
                    crystal_type = h5f["GlobalLabels"]["CrystalType"][()].decode()

                    # Space group
                    space_group_symbol = h5f["GlobalLabels"]["SpaceGroupSymbol"][()].decode()
                    space_group_number = h5f["GlobalLabels"]["SpaceGroupNumber"][()]

                    # Crystal system
                    crystal_system = h5f["GlobalLabels"]["CrystalSystem"][()].decode()
                    crystal_system_number = self.crystal_system_to_number(crystal_system)

                    # Loop through all particle sizes
                    for key in h5f["DiscreteParticleGraphs"].keys():
                        node_feat = torch.tensor(h5f["DiscreteParticleGraphs"][key]["NodeFeatures"][:], dtype=torch.float32)
                        edge_index = torch.tensor(h5f["DiscreteParticleGraphs"][key]["EdgeDirections"][:],dtype=torch.long)
                        edge_attr = torch.tensor(h5f["DiscreteParticleGraphs"][key]["EdgeFeatures"][:], dtype=torch.float32)
                        pos_abs = torch.tensor(h5f["DiscreteParticleGraphs"][key]["AbsoluteCoordinates"][:], dtype=torch.float32)
                        pos_frac = torch.tensor(h5f["DiscreteParticleGraphs"][key]["FractionalCoordinates"][:], dtype=torch.float32)

                        # Create graph data object
                        data = Data(
                            data_id = raw_path.split("/")[-1].split(".")[0],
                            x = node_feat,
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos_abs = pos_abs,
                            pos_frac = pos_frac,
                            
                            y=dict(
                                crystal_type=crystal_type,
                                space_group_symbol=space_group_symbol,
                                space_group_number=space_group_number,
                                crystal_system=crystal_system,
                                crystal_system_number=crystal_system_number,
                                atomic_species=atomic_species,#.unsqueeze(0),
                                n_atomic_species=len(atomic_species),
                                np_size=h5f["DiscreteParticleGraphs"][key]["NP size (Å)"][()],
                                n_atoms=node_feat.shape[0],
                                n_bonds=edge_index.shape[1],

                                cell_params=cell_params.unsqueeze(0),
                                unit_cell_x=unit_cell_node_feat,
                                unit_cell_edge_index=unit_cell_edge_index,
                                unit_cell_edge_attr=unit_cell_edge_attr,
                                unit_cell_pos_abs=unit_cell_pos_abs,
                                unit_cell_pos_frac=unit_cell_pos_frac,
                                unit_cell_n_atoms=unit_cell_node_feat.shape[0],
                                unit_cell_n_bonds=unit_cell_edge_index.shape[1],

                                # Scattering data
                                nd=torch.tensor(h5f["ScatteringData"][key]["ND"][:], dtype=torch.float32).unsqueeze(0),
                                xrd=torch.tensor(h5f["ScatteringData"][key]["XRD"][:], dtype=torch.float32).unsqueeze(0),
                                nPDF=torch.tensor(h5f["ScatteringData"][key]["nPDF"][:], dtype=torch.float32).unsqueeze(0),
                                xPDF=torch.tensor(h5f["ScatteringData"][key]["xPDF"][:], dtype=torch.float32).unsqueeze(0),
                                sans=torch.tensor(h5f["ScatteringData"][key]["SANS"][:], dtype=torch.float32).unsqueeze(0),
                                saxs=torch.tensor(h5f["ScatteringData"][key]["SAXS"][:], dtype=torch.float32).unsqueeze(0),
                            ),
                        )
                        
                        # Pad tensors to match the largest unit cell size in the dataset
                        unit_cell_node_feat_padded = torch.nn.functional.pad(unit_cell_node_feat, (0, 0, 0, max_unit_cell_n_atoms - unit_cell_node_feat.shape[0]), "constant", 0)
                        unit_cell_pos_abs_padded = torch.nn.functional.pad(unit_cell_pos_abs, (0, 0, 0, max_unit_cell_n_atoms - unit_cell_node_feat.shape[0]), "constant", 0)
                        unit_cell_pos_frac_padded = torch.nn.functional.pad(unit_cell_pos_frac, (0, 0, 0, max_unit_cell_n_atoms - unit_cell_node_feat.shape[0]), "constant", -1)
                        # Create unit cell graph data object
                        data_unit_cell = Data(
                            data_id = raw_path.split("/")[-1].split(".")[0],
                            x = unit_cell_node_feat_padded,
                            edge_index = unit_cell_edge_index,
                            edge_attr = unit_cell_edge_attr,
                            pos_abs = unit_cell_pos_abs_padded,
                            pos_frac = unit_cell_pos_frac_padded,
                            
                            y = dict(
                                crystal_type=crystal_type,
                                space_group_symbol=space_group_symbol,
                                space_group_number=space_group_number,
                                crystal_system=crystal_system,
                                crystal_system_number=crystal_system_number,
                                atomic_species=atomic_species,
                                n_atomic_species=len(atomic_species),
                                np_size=h5f["DiscreteParticleGraphs"][key]["NP size (Å)"][()],
                                n_atoms=unit_cell_node_feat.shape[0],
                                n_bonds=unit_cell_edge_index.shape[1],
                                
                                cell_params=cell_params.unsqueeze(0),
                                
                                # Scattering data
                                nd=torch.tensor(h5f["ScatteringData"][key]["ND"][:], dtype=torch.float32).unsqueeze(0),
                                xrd=torch.tensor(h5f["ScatteringData"][key]["XRD"][:], dtype=torch.float32).unsqueeze(0),
                                nPDF=torch.tensor(h5f["ScatteringData"][key]["nPDF"][:], dtype=torch.float32).unsqueeze(0),
                                xPDF=torch.tensor(h5f["ScatteringData"][key]["xPDF"][:], dtype=torch.float32).unsqueeze(0),
                                sans=torch.tensor(h5f["ScatteringData"][key]["SANS"][:], dtype=torch.float32).unsqueeze(0),
                                saxs=torch.tensor(h5f["ScatteringData"][key]["SAXS"][:], dtype=torch.float32).unsqueeze(0),
                            ),
                        )
                        # Create graph of n most central atoms (n = 100 maximum)
                        # Calculate distance from origo to each atom
                        dist = torch.norm(pos_abs, dim=1)
                        # Sort atoms by distance from origo
                        sorted_idx = torch.argsort(dist)

                        # Only create central atom graphs for structures with at least 100 atoms
                        if len(pos_abs) >= 100:
                            # Take exactly 100 most central atoms
                            selected_idx = sorted_idx[:100]

                            # Select 100 most central atoms
                            central_node_feat = node_feat[selected_idx]
                            central_pos_abs = pos_abs[selected_idx]
                            
                            # Select subgraph containing central atoms
                            central_edge_index, central_edge_attr = subgraph(selected_idx, edge_index, edge_attr, relabel_nodes=True)
                            
                            # Get fractional coordinates relative to the unit cell parameters
                            atom_obj = Atoms(
                                symbols = central_node_feat[:, 0].numpy(),
                                positions = central_pos_abs.numpy(),
                                cell = cell_params.numpy(),
                            )
                            
                            central_pos_frac = torch.tensor(atom_obj.get_scaled_positions(), dtype=torch.float32)
                            data_central = Data(
                                data_id = raw_path.split("/")[-1].split(".")[0],
                                x = central_node_feat,
                                edge_index = central_edge_index,
                                edge_attr = central_edge_attr,
                                pos_abs = central_pos_abs,
                                pos_frac = central_pos_frac,
                                
                                y=dict(
                                    crystal_type=crystal_type,
                                    space_group_symbol=space_group_symbol,
                                    space_group_number=space_group_number,
                                    crystal_system=crystal_system,
                                    crystal_system_number=crystal_system_number,
                                    atomic_species=atomic_species,#.unsqueeze(0),
                                    n_atomic_species=len(atomic_species),
                                    np_size=h5f["DiscreteParticleGraphs"][key]["NP size (Å)"][()],
                                    n_atoms=central_node_feat.shape[0],
                                    n_bonds=central_edge_index.shape[1],

                                    cell_params=cell_params.unsqueeze(0),
                                    unit_cell_x=unit_cell_node_feat,
                                    unit_cell_edge_index=unit_cell_edge_index,
                                    unit_cell_edge_attr=unit_cell_edge_attr,
                                    unit_cell_pos_abs=unit_cell_pos_abs,
                                    unit_cell_pos_frac=unit_cell_pos_frac,
                                    unit_cell_n_atoms=unit_cell_node_feat.shape[0],
                                    unit_cell_n_bonds=unit_cell_edge_index.shape[1],

                                    # Scattering data
                                    nd=torch.tensor(h5f["ScatteringData"][key]["ND"][:], dtype=torch.float32).unsqueeze(0),
                                    xrd=torch.tensor(h5f["ScatteringData"][key]["XRD"][:], dtype=torch.float32).unsqueeze(0),
                                    nPDF=torch.tensor(h5f["ScatteringData"][key]["nPDF"][:], dtype=torch.float32).unsqueeze(0),
                                    xPDF=torch.tensor(h5f["ScatteringData"][key]["xPDF"][:], dtype=torch.float32).unsqueeze(0),
                                    sans=torch.tensor(h5f["ScatteringData"][key]["SANS"][:], dtype=torch.float32).unsqueeze(0),
                                    saxs=torch.tensor(h5f["ScatteringData"][key]["SAXS"][:], dtype=torch.float32).unsqueeze(0),
                                ),
                            )
                            
                            # Save central atom data
                            torch.save(data_central, os.path.join(self.processed_dir + '_central', f"data_{idx}.pt"))
                        else:
                            # Skip creating central atom graph for structures with fewer than 100 atoms
                            # Create an empty file to maintain indexing consistency
                            with open(os.path.join(self.processed_dir + '_central', f"data_{idx}.pt.skipped"), 'w') as f:
                                f.write(f"Skipped: Structure has only {len(pos_abs)} atoms, fewer than required 100")

                        # Apply filters
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        # Apply transforms
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                            data_unit_cell = self.pre_transform(data_unit_cell)
                            data_central = self.pre_transform(data_central)
                            # data_super_cell = self.pre_transform(data_super_cell)
                            # data_combi = self.pre_transform(data_combi)

                        # Save to `self.processed_dir`.
                        torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                        torch.save(data_unit_cell, os.path.join(self.processed_dir + '_unit_cell', f"data_{idx}.pt"))

                        # Update index
                        idx += 1

                # Update process pbar
                process_pbar.update(1)

            except Exception as e:
                self.write_to_log('processing_error_log.out', str(raw_path) + '\n' + str(e) + '\n')

        process_pbar.close()

    def len(
        self,
        split: Optional[str] = None
    ) -> int:
        """
        Returns the length of the dataset.

        Args:
            split (str, optional): Type of split. Defaults to None.

        Returns:
            int: Length of the dataset.
        
        Raises:
            ValueError: If split is not recognized.
        """
        if split is None:
            length = len(self.processed_file_names)
        elif split.lower() == "train":
            length = len(self.train_set)
        elif split.lower() in ["validation", "val"]:
            length = len(self.validation_set)
        elif split.lower() == "test":
            length = len(self.test_set)
        else:
            raise ValueError(
                'Split not recognized. Please use either "train", "validation" or "test"'
            )
        return length

    def get(
        self,
        idx: int,
        split: Optional[str] = None,
    ) -> Data:
        """
        Returns the data at the given index.

        Args:
            idx (int): Index of the data.
            split (str, optional): Type of split. Defaults to None.

        Returns:
            Data: Data object at the given index.
        
        Raises:
            ValueError: If split is not recognized.
        """
        if split is None:
            if self.graph_type == "unit_cell":
                data = torch.load(os.path.join(self.processed_dir + '_unit_cell', f"data_{idx}.pt"), weights_only=False)
            elif self.graph_type == "central":
                data = torch.load(os.path.join(self.processed_dir + '_central', f"data_{idx}.pt"), weights_only=False)
            else:
                data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False)
        elif split.lower() == "train":
            data = self.train_set[idx]
        elif split.lower() in ["validation", "val"]:
            data = self.validation_set[idx]
        elif split.lower() == "test":
            data = self.test_set[idx]
        else:
            raise ValueError(
                'Split not recognized. Please use either "train", "validation" or "test"'
            )
        return data

    def create_data_split(
        self,
        test_size: float = 0.1,
        validation_size: Optional[float] = None,
        split_strategy: str = "random",
        stratify_on: str ="Space group (Number)",
        stratify_distribution: str = "match",
        n_samples_per_class: str = "max",
        random_state: int = 42,
        return_idx: bool = False,
    ) -> Optional[Union[List[int], None]]:
        """
        Split the dataset into train, validation and test sets. The indices of the split are saved to csv files in the processed directory.
        The same split is applied to all three graph types (full, unit_cell, central).
        """

        if validation_size is None:
            validation_size = test_size

        # Get statistics for all graph types
        df_stats_all = self.get_statistics(return_dataframe=True)
        df_stats = df_stats_all["full"]  # Use full graph statistics for splitting

        if split_strategy == "random":
            # Split data into train, validation and test sets
            train_idx, test_idx = train_test_split(
                np.arange(self.len()),
                test_size = test_size,
                random_state = random_state
            )
            train_idx, validation_idx = train_test_split(
                train_idx,
                test_size = validation_size / (1 - test_size),
                random_state = random_state
            )

            # Save indices to csv files (only once)
            np.savetxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_train.csv"),
                train_idx,
                delimiter=",",
                fmt="%i",
            )
            np.savetxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_validation.csv"),
                validation_idx,
                delimiter=",",
                fmt="%i",
            )
            np.savetxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_test.csv"),
                test_idx,
                delimiter=",",
                fmt="%i",
            )

            # Update statistics dataframe for all graph types
            for graph_type, df in df_stats_all.items():
                df[f"{split_strategy.capitalize()} data split"] = ""
                df[f"{split_strategy.capitalize()} data split"].loc[train_idx] = "Train"
                df[f"{split_strategy.capitalize()} data split"].loc[validation_idx] = "Validation"
                df[f"{split_strategy.capitalize()} data split"].loc[test_idx] = "Test"

        elif split_strategy == "stratified":
            if stratify_distribution == "match":
                # Split data into train, validation and test sets
                train_idx, test_idx = train_test_split(
                    np.arange(self.len()),
                    test_size=test_size,
                    random_state=random_state,
                    stratify=df_stats[stratify_on],
                )
                train_idx, validation_idx = train_test_split(
                    train_idx,
                    test_size=validation_size / (1 - test_size),
                    random_state=random_state,
                    stratify=df_stats.loc[train_idx][stratify_on],
                )

                # Save indices to csv files (only once)
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_train.csv'),
                    train_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_validation.csv'),
                    validation_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_test.csv'),
                    test_idx,
                    delimiter=",",
                    fmt="%i",
                )

                # Update statistics dataframe for all graph types
                for graph_type, df in df_stats_all.items():
                    df[f"{split_strategy.capitalize()} data split ({stratify_on})"] = ""
                    df[f"{split_strategy.capitalize()} data split ({stratify_on})"].loc[train_idx] = "Train"
                    df[f"{split_strategy.capitalize()} data split ({stratify_on})"].loc[validation_idx] = "Validation"
                    df[f"{split_strategy.capitalize()} data split ({stratify_on})"].loc[test_idx] = "Test"

            elif stratify_distribution == "equal":
                if n_samples_per_class == "max":
                    # Find the class with the least number of samples
                    min_samples = df_stats[stratify_on].value_counts().min()
                elif isinstance(n_samples_per_class, int):
                    min_samples = n_samples_per_class
                else:
                    raise ValueError(
                        'n_samples_per_class not recognized. Please use either "max" or an integer'
                    )

                # Randomly sample the same number of samples from each class
                subset_idx = []
                for group in df_stats[stratify_on].unique():
                    subset_idx += list(
                        df_stats[df_stats[stratify_on] == group]
                        .sample(min_samples, random_state=random_state)
                        .index
                    )

                # Find the total number of samples
                n_samples = len(subset_idx)

                # Find the number of samples to use for train, validation and test sets
                n_test = int(n_samples * test_size)
                n_validation = int((n_samples - n_test) * validation_size / (1 - test_size))
                n_train = n_samples - n_test - n_validation

                # Split data into train, validation and test sets
                train_idx, test_idx = train_test_split(
                    subset_idx,
                    train_size = n_train + n_validation,
                    test_size = n_test,
                    random_state = random_state,
                    stratify = df_stats.loc[subset_idx][stratify_on],
                )
                train_idx, validation_idx = train_test_split(
                    train_idx,
                    train_size = n_train,
                    test_size = n_validation,
                    random_state = random_state,
                    stratify = df_stats.loc[train_idx][stratify_on],
                )

                # Save indices to csv files (only once)
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_train.csv'),
                    train_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_validation.csv'),
                    validation_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_test.csv'),
                    test_idx,
                    delimiter=",",
                    fmt="%i",
                )

                # Update statistics dataframe for all graph types
                for graph_type, df in df_stats_all.items():
                    df[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"] = ""
                    df[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"].loc[train_idx] = "Train"
                    df[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"].loc[validation_idx] = "Validation"
                    df[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"].loc[test_idx] = "Test"

        else:
            raise ValueError(
                'Split strategy not recognized. Please use either "random" or "stratified"'
            )

        # Save updated statistics for all graph types
        for graph_type, df in df_stats_all.items():
            suffix = f"_{graph_type}" if graph_type else ""
            df.to_pickle(os.path.join(self.root, f"dataset_statistics{suffix}.pkl"))

        if return_idx:
            return train_idx, validation_idx, test_idx
        else:
            return None

    def load_data_split(
        self,
        split_strategy: str = "random",
        stratify_on: str = "Space group (Number)",
        stratify_distribution: str = "match",
    ) -> None:
        """
        Load the indices of the train, validation and test sets from csv files in the processed directory.

        Args:
            split_strategy (str, optional): Split strategy. Defaults to "random".
            stratify_on (str, optional): Feature to stratify on. Defaults to "Space group (Number)".
            stratify_distribution (str, optional): Distribution of stratification. Defaults to "match".
        """
        if split_strategy == "random":

            # Load indices from csv files
            train_idx = np.loadtxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_train.csv"),
                delimiter=",",
                dtype=int,
            )
            validation_idx = np.loadtxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_validation.csv"),
                delimiter=",",
                dtype=int,
            )
            test_idx = np.loadtxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_test.csv"),
                delimiter=",",
                dtype=int,
            )

        elif split_strategy == "stratified":

            if stratify_distribution == "match":

                # Load indices from csv files
                train_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_train.csv'),
                    delimiter=",",
                    dtype=int,
                )
                validation_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_validation.csv'),
                    delimiter=",",
                    dtype=int,
                )
                test_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_test.csv'),
                    delimiter=",",
                    dtype=int,
                )

            elif stratify_distribution == "equal":

                # Load indices from csv files
                train_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_train.csv'),
                    delimiter=",",
                    dtype=int,
                )
                validation_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_validation.csv'),
                    delimiter=",",
                    dtype=int,
                )
                test_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_test.csv'),
                    delimiter=",",
                    dtype=int,
                )

        # Split the dataset into train, validation and test sets
        self.train_set = Subset(self, train_idx)
        self.validation_set = Subset(self, validation_idx)
        self.test_set = Subset(self, test_idx)

    def get_statistics(
        self,
        return_dataframe: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Computes statistics of the dataset for all graph types.

        Args:
            return_dataframe (bool, optional): Whether to return a dataframe. Defaults to False.

        Returns:
            pd.DataFrame: Statistics of the dataset.
        """
        # Get statistics paths for all three types
        stat_paths = {
            "": os.path.join(self.root, "dataset_statistics.pkl"),
            "unit_cell": os.path.join(self.root, "dataset_statistics_unit_cell.pkl"),
            "central": os.path.join(self.root, "dataset_statistics_central.pkl")
        }

        df_stats_all = {}
        
        for graph_type, stat_path in stat_paths.items():
            # Read pkl or generate
            if os.path.exists(stat_path):
                df_stats_all[graph_type] = pd.read_pickle(stat_path)
            else:
                df_stats = pd.DataFrame(
                    columns=[
                        "idx",
                        "# of nodes",
                        "# of edges",
                        "edge/node ratio",
                        "# of elements",
                        "Space group (Symbol)",
                        "Space group (Number)",
                        "Crystal type",
                        "Crystal system",
                        "Crystal system (Number)",
                        "NP size (Å)",
                        "Elements",
                    ]
                )

                # Store current graph type
                original_graph_type = self.graph_type
                # Temporarily set graph type
                self.graph_type = graph_type

                stat_pbar = tqdm(desc=f'Generating statistics for {graph_type if graph_type else "full"} graph...', 
                               total=self.len(), leave=False)
                
                for idx in range(self.len()):
                    try:
                        graph = self.get(idx=idx)
                        df_stats.loc[df_stats.shape[0]] = [
                            idx,
                            float(graph.num_nodes),
                            float(graph.num_edges),
                            float(graph.num_edges) / float(graph.num_nodes),
                            float(graph.y["n_atomic_species"]),
                            graph.y["space_group_symbol"],
                            float(graph.y["space_group_number"]),
                            graph.y["crystal_type"],
                            graph.y["crystal_system"],
                            graph.y["crystal_system_number"],
                            graph.y["np_size"],
                            graph.y["atomic_species"],
                        ]
                    except Exception as e:
                        print(f"Error processing idx {idx} for {graph_type} graph: {str(e)}")
                    stat_pbar.update(1)
                stat_pbar.close()

                # Save statistics
                df_stats.to_pickle(stat_path)
                df_stats_all[graph_type] = df_stats
                
                # Restore original graph type
                self.graph_type = original_graph_type

        if return_dataframe:
            if self.train_set is not None:
                indices = (list(self.train_set.indices) + 
                          list(self.validation_set.indices) + 
                          list(self.test_set.indices))
                return {
                    "full": df_stats_all[""].loc[indices].reset_index(drop=True),
                    "unit_cell": df_stats_all["unit_cell"].loc[indices].reset_index(drop=True),
                    "central": df_stats_all["central"].loc[indices].reset_index(drop=True)
                }
            else:
                return {
                    "full": df_stats_all[""],
                    "unit_cell": df_stats_all["unit_cell"],
                    "central": df_stats_all["central"]
                }
        else:
            return None