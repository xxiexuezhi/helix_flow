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
from torch_cluster import knn_graph

class ProteinDataset(Dataset):
    def __init__(self, dataset_path, full_atom=False, use_epitope=False, mask_type='cdrh3', edge_type='fc'):
        self.full_atom = full_atom
        self.use_epitope = use_epitope
        self.mask_type = mask_type
        self.edge_type = edge_type

        with open(dataset_path, 'rb') as f:
            self.structures = pkl.load(f)

        self.structures = [i for i in self.structures if (i['h_chain']['cdr_mask'] == 3).sum() > 0] # remove 0 length cdrh3

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

        # combine all chains
        aa, all_coord, all_mask, cdr_mask = [], [], [], []
        aa.append(structure['h_chain']['aa'])
        all_coord.append(structure['h_chain']['coord'])
        all_mask.append(structure['h_chain']['atom_mask'])
        cdr_mask.append(structure['h_chain']['cdr_mask'])

        if structure['l_chain'] is not None:
            aa.append(structure['l_chain']['aa'])
            all_coord.append(structure['l_chain']['coord'])
            all_mask.append(structure['l_chain']['atom_mask'])
            cdrl_mask = structure['l_chain']['cdr_mask'] if self.mask_type == 'all' else torch.zeros_like(structure['l_chain']['cdr_mask'])
            cdr_mask.append(cdrl_mask)

        if self.use_epitope:
            epitope_mask = structure['ag_chain']['epitope_mask'].bool()
            aa_ag = structure['ag_chain']['aa']
            X_ag = structure['ag_chain']['coord']
            X_epi = X_ag[epitope_mask]
            aa_epi = aa_ag[epitope_mask]
            atom_mask = structure['ag_chain']['atom_mask'][epitope_mask]
            aa.append(aa_epi)
            all_coord.append(X_epi)
            all_mask.append(atom_mask)
            cdr_mask.append(torch.zeros(X_epi.shape[0]))
        else:
            aa.append(structure['ag_chain']['aa'])
            all_coord.append(structure['ag_chain']['coord'])
            all_mask.append(structure['ag_chain']['atom_mask'])
            cdr_mask.append(structure['ag_chain']['cdr_mask'])

        aa = torch.cat(aa, dim=0)
        all_coord = torch.cat(all_coord,dim=0)
        all_mask = torch.cat(all_mask, dim=0)
        cdr_mask = torch.cat(cdr_mask, dim=0)

        # aa to tensor
        aa_onehot = F.one_hot(aa,num_classes=21).float()

        # generate edge index
        if self.edge_type == 'fc':
            edge_index = []
            length = len(aa)
            for i in range(length):
                for j in range(length):
                    if i == j: continue
                    edge_index.append([i,j])
                    edge_index.append([j,i])
            edge_index = torch.tensor(edge_index).long().T
        elif self.edge_type == 'knn':
            edge_index = knn_graph(all_coord[:,1], k=30)

        origin = all_coord[:,:1].reshape(-1, 3).mean(0)
        all_coord = (all_coord - origin.view(1,1,-1)) * all_mask.unsqueeze(-1)

        data = Data(edge_index=edge_index, aa=aa, aa_onehot=aa_onehot, id=structure['id'],
                 pos=all_coord, atom_mask=all_mask, cdr_mask=cdr_mask)

        return data

    def __len__(self):
        return len(self.structures)

def get_dataloader(config, sample=False, ddp=False):
    train_ds = ProteinDataset(config.data.train_path, use_epitope=config.data.use_epitope)
    # test_ds = ProteinDataset(config.data.test_path)

    batch_size = config.train.batch_size if not sample else config.sample.batch_size

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        # test_sampler = DistributedSampler(test_ds)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
        # test_dl = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler)
        return train_dl, None, train_sampler, None
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        # test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        return train_dl, None, None, None