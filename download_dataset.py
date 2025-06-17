# Test script for downloading dataset and making splits
import warnings
from CHILI_centralAtoms import CHILI

root = 'data'
dataset='CHILI-3K'
dataset = CHILI(root, dataset,graph_type='central', pre_transform=None, pre_filter=None)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    dataset.create_data_split(split_strategy = 'random', test_size=0.1)