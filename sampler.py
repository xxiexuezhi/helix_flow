import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.print_colors import red
from utils.loader import load_seed, load_device, load_model_params, \
    load_ema
from utils.logger import Logger, set_log, start_log
from models.nets.graph_attention_transformer import GraphAttentionTransformer
from utils.train_utils import count_parameters
from pathlib import Path
from dataset import get_dataloader
from utils.structure_utils import create_structure_from_crds, count_clashes
from utils.sidechain_utils import VirtualToAllAtomCoords
from loss import FlowMatchingLoss
from models.cnf import CNF


class Trainer3D(object):
    def __init__(self, config, ddp=False):
        super(Trainer3D, self).__init__()
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader, _, _ = get_dataloader(self.config, ddp=ddp)

        self.params = load_model_params(self.config)
        self.virtual_to_all_atom = VirtualToAllAtomCoords(use_native_bb_coords=True, virtual=True)

    def train(self, ts, resume=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        # -------- Load models, optimizers, ema --------
        use_saprot = True if self.config.data.train.saprot is not None else False
        self.model = CNF(GraphAttentionTransformer(**self.config.model, saprot_emb=use_saprot).cuda())
        # self.model = torch.compile(self.model)
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        print(red(f'{self.ckpt}'))

        ckpt_dict = torch.load(
            f'./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.config.ckpt}.pth')
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.ema.load_state_dict(ckpt_dict['ema'])
        print(f'Loaded checkpoint {self.config.ckpt}')

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)

        save_path = Path(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath('samples')
        sample_path.mkdir(exist_ok=True, parents=True)

        self.model.eval()
        self.ema.copy_to(self.model.parameters())

        with torch.no_grad():
            atom_rmsd, num_clashes = {}, {}
            for batch in self.test_loader:
                batch = batch.to(f'cuda:{self.device[0]}')
                aa_str, aa_onehot, aa_num, atom_type, coords, mask, atom_mask, batch_id, pdb_codes = batch.aa_str, batch.aa_onehot, batch.aa, batch.atom_type, \
                                                                                                     batch.pos, batch.aa_mask, batch.atom_mask, batch.batch, batch.id
                bb_coords = coords[:, :4]

                for i in pdb_codes:
                    atom_rmsd[i] = []
                    num_clashes[i] = []

                outPath = sample_path.joinpath(f'{self.ckpt}')
                outPath.mkdir(exist_ok=True, parents=True)

                if self.config.model.add_atom_type:
                    aa_onehot = torch.cat([aa_onehot, atom_type.view(atom_type.shape[0], -1)], dim=-1)

                batch_size = batch_id.max().item() + 1

                for r in range(self.config.sample.n_samples):
                    with torch.no_grad():
                        # Sampling
                        z = torch.randn(bb_coords.shape[0], 5, 3).to(self.device[0])
                        pred_sc = self.model.decode(z, batch)
                        likelihood = self.model.log_prob(pred_sc, batch)

                        bb_coords = bb_coords / self.config.data.scale_coords
                        pred_sc = pred_sc / self.config.data.scale_coords
                        all_atom_coords = self.virtual_to_all_atom(aa_num, bb_coords, pred_sc)
                        coords = coords / self.config.data.scale_coords  # scale back for rmsd calculation

                        # Save to PDB
                        for i in range(batch_size):
                            crds_batch = coords[batch_id == i]
                            pos_batch = all_atom_coords[batch_id == i]
                            mask_batch = mask[batch_id == i]
                            atom_mask_batch = atom_mask[batch_id == i]

                            # evaluate atom-level RMSD and clash
                            rmsd = torch.linalg.norm(pos_batch[:, 4:] - crds_batch[:, 4:], dim=-1)
                            rmsd = (rmsd * atom_mask_batch[:, 4:]).sum(-1) / (
                                        atom_mask_batch[:, 4:].sum(-1) + 1e-8)
                            atom_rmsd[pdb_codes[i]].append(rmsd.mean().item())

                            pdb_path = outPath.joinpath(batch['id'][i])
                            pdb_path.mkdir(exist_ok=True, parents=True)
                            create_structure_from_crds(aa_str[i], pos_batch.cpu(), mask_batch.cpu(),
                                                       outPath=str(pdb_path.joinpath(f'run_{r + 1}.pdb')))

                            num_clashes[pdb_codes[i]].append(count_clashes(
                                pdb_path.joinpath(f'run_{r + 1}.pdb')))  # TODO: terribly inefficient

        torch.save({'rmsd': atom_rmsd, 'clash': num_clashes}, outPath.joinpath('raw_stats.pth'))

        self.ema.restore(self.model.parameters())

        print(' ')
        return self.ckpt


if __name__ == '__main__':
    import argparse
    from parsers.config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = get_config(args.config, seed=args.seed)
    trainer = Trainer3D(config)
    trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()), resume=args.resume)
