import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import multiprocessing
import logging
import numpy as np
import random
from tqdm.contrib.concurrent import process_map

import warnings
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure

from utils.constants import three_to_one_letter, non_standard_to_standard, letter_to_num, max_num_heavy_atoms, restype_to_heavyatom_names, heavyatom_to_label
from torch_geometric.data import Data, DataLoader
import pickle as pkl

class ProteinDataset(Dataset):
    def __init__(self, dataset_path, scale_coords=1.0):
        self.scale_coords = scale_coords
        with open(dataset_path, 'rb') as f:
            self.structures = pkl.load(f)

        # Remove None from self.structures
        self.structures = [self.to_tensor(i) for i in self.structures if i is not None]

        print(f'Loaded {len(self.structures)} data points...')

    def to_tensor(self, d, exclude=[]):
        feat_dtypes = {
            "id": None,
            "chain": None,
            "coord": torch.float32,
            "origin": torch.float32,
            "atom_type": torch.long,
            "aa": torch.long,
            "aa_mask": torch.long,
            "atom_mask": torch.float32,
            "cdr_mask": torch.float32,
            "epitope_mask": torch.float32
        }

        for x in exclude:
            del d[x]

        for k,v in d.items():
            if type(v) == dict:
                d[k] = self.to_tensor(v)
            elif type(v) == list or type(v) == np.ndarray:
                if feat_dtypes[k] is not None:
                    d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])

        return d

    def __getitem__(self, idx):
        structure = self.structures[idx]

        # aa to tensor
        seq_num = structure['aa']
        seq_onehot = F.one_hot(seq_num,num_classes=21).float()
        coords = structure['coord']
        atom_mask = structure['atom_mask']
        aa_mask = structure['aa_mask']

        # generate edge index
        edge_index = []
        length = len(seq_num)
        for i in range(length):
            for j in range(length):
                if i == j: continue
                edge_index.append([i,j])
                edge_index.append([j,i])
        edge_index = torch.tensor(edge_index).long().T

        origin = coords[:,:4].reshape(-1, 3).mean(0)
        coords = (coords - origin.view(1,1,-1)) * atom_mask.unsqueeze(-1)
        coords = coords * self.scale_coords

        data = Data(edge_index=edge_index, aa=seq_num, aa_onehot=seq_onehot,id=str(idx),# id=structure['id'], update heredue to helix data has no id.
                 pos=coords, aa_mask=aa_mask, atom_mask=atom_mask)

        return data

    def __len__(self):
        return len(self.structures)

def get_dataloader(config, sample=False, ddp=False):
    train_ds = ProteinDataset(config.data.train_path, scale_coords=config.data.scale_coords)
    test_ds = ProteinDataset(config.data.test_path, scale_coords=config.data.scale_coords)
    batch_size = config.train.batch_size if not sample else config.sample.batch_size

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
        test_dl = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler)
        return train_dl, test_dl, train_sampler, test_sampler
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        return train_dl, test_dl, None, None
