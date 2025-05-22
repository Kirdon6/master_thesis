import warnings
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN
from CHILI_centralAtoms import CHILI

root = 'data'
dataset='CHILI-3K'
dataset = CHILI(root, dataset,graph_type='central', pre_transform=None, pre_filter=None)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    dataset.create_data_split(split_strategy = 'random', test_size=0.1)