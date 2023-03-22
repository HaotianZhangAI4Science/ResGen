import pickle
import torch
import os
import numpy as np
from torch.utils.data import Subset
from utils.datasets.pl import PocketLigandPairDataset
from utils.datasets.resgen import ResGenDataset 


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset

def transform_data(data, transform):
    assert data.protein_pos.size(0) > 0
    if transform is not None:
        data = transform(data)
    return data