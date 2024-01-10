import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#from utils.print_colors import red
from utils.loader import load_seed, load_device, load_ema#, load_model_params
from utils.logger import Logger, set_log, start_log
from models.nets.graph_attention_transformer_aa import GraphAttentionTransformer
from utils.train_utils import count_parameters
from pathlib import Path
from dataset import get_dataloader
from utils.structure_utils import create_structure_from_crds
from loss2 import FlowMatchingLoss
from models.cnf2 import CNF


from utils.constants import num_to_letter

import pickle

#from utils.loader2 import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
#                         load_ema, load_e3nn_loss_fn, load_ema_from_ckpt, \
#                         load_ckpt, load_opt_from_ckpt, load_model_from_ckpt

use_gpu = False

class Gnereatate3D(object):
    def __init__(self, config,ddp=False):
        super(Gnereatate3D, self).__init__()
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader, _, _ = get_dataloader(self.config,ddp=ddp)
        #self.cpt =
        #self.params = load_model_params(self.config)

    def g(self, ts, int_cpt):
        self.config.exp_name = ts
        self.ckpt = str(int_cpt)
        
        # -------- Load models, optimizers, ema --------
        if use_gpu:
            self.model = CNF(GraphAttentionTransformer(**self.config.model).cuda())
        else:
            self.model = CNF(GraphAttentionTransformer(**self.config.model))

        # self.model = torch.compile(self.model)
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr,
                                    weight_decay=self.config.train.weight_decay, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.train.lr_decay)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        #print(red(f'{self.ckpt}'))

        if self.config.ckpt is not None:
            self.config.ckpt  = self.ckpt
            #print('./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.config.ckpt}.pth')
            if use_gpu:
                ckpt_dict = torch.load(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.config.ckpt}.pth')
            else:
                ckpt_dict = torch.load(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.config.ckpt}.pth',map_location=torch.device('cpu') )
            self.model.load_state_dict(ckpt_dict['state_dict'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.ema.load_state_dict(ckpt_dict['ema'])
            start_epoch = int(self.config.ckpt.split("_")[-1])
            print(f'Loaded checkpoint {self.config.ckpt}')
            epoch = self.ckpt
        else:
            PRINT("NOT LOADED")
            start_epoch = 0

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)

        writer = SummaryWriter(os.path.join(*['logs_train', 'tensorboard', self.config.data.data,
                                            self.config.train.name, self.config.exp_name]))

        save_path = Path(f'./samples/')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path#.joinpath('samples')
        #sample_path.mkdir(exist_ok=True, parents=True)

        self.loss_fn = FlowMatchingLoss(self.model, self_cond=self.config.model.self_cond)

        train_pos_iter = []
        seq_lst = []
        num_iter = 0
        write_to_tb = False # used to handle writing to tensorboard with gradient accumulation
        # -------- Training --------
        if  True:
        #epoch % self.config.train.sample_interval == self.config.train.sample_interval-1:
            if True:        # -------- Generate samples --------
                self.model.eval()
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model.parameters())

                with torch.no_grad():
                    for batch in self.train_loader:
                        if use_gpu:
                            batch = batch.to(f'cuda:{self.device[0]}')
                        else:
                            self.device = ["cpu"]
                            batch = batch.to(self.device[0])
                        coords, mask, atom_mask, batch_id = batch.pos, batch.aa_mask, batch.atom_mask, batch.batch
                        
                        aa = batch.aa_onehot

                        bb_coords = coords[:, :14]

                        outPath = sample_path.joinpath( f'epoch_{epoch}')
                        outPath.mkdir(exist_ok=True, parents=True)

                        batch_size = batch_id.max().item() + 1

                        for r in range(self.config.sample.n_samples):
                            with torch.no_grad():
                                # Sampling
                                z = torch.randn(*bb_coords.shape).to(self.device[0])
                                

                                aa = batch.aa_onehot

                                z_aa = torch.randn_like(aa).to(self.device[0])
 
                                #z = torch.randn(bb_coords.shape[0],7+4,3).to(self.device[0])


                                pred_aa,pred_pos = self.model.decode(z_aa, z, batch)

                                #pred_aa = pred_all[:,:7,:]

                                #pred_aa = pred_aa.view(-1,21)

                                #3pred_pos = pred_all[:,7:,:]



                                #pred_aa, pred_pos = self.model.decode(z_aa, z, batch)
                                pred_pos = pred_pos / self.config.data.scale_coords

                                # Save to PDB
                                for i in range(batch_size):
                                    pos_batch = pred_pos[batch_id == i]
                                    mask_batch = mask[batch_id == i]
                                    #aa = "G"*pos_batch.shape[0] # set to all glycines for now
                                    seq = pred_aa[batch_id == i]
                                    seq = torch.argmax(seq, dim=-1)
                                    m = torch.ones_like(seq)
                                    seq = ''.join([num_to_letter[h.item()] for h in seq])
                                    aa = seq



                                    pdb_path = outPath#.joinpath(batch['id'][i])
                                    #pdb_path.mkdir(exist_ok=True, parents=True)
                                    
                                    #with open("g"+str(epoch)+"_sample444.pkl","wb") as fout:
                                    #    seq_lst.append([epoch, aa, pred_pos.cpu(), pred_aa.cpu(), pos_batch.cpu(), mask_batch.cpu()])
                                    #    pickle.dump(seq_lst,fout)

                                    #aa = "G"*pos_batch.shape[0]

                                    create_structure_from_crds(aa, pos_batch.cpu(), mask_batch.cpu(),
                                                               outPath=str(pdb_path.joinpath(f'run_{r + 1}.pdb')))
                        break
                self.ema.restore(self.model.parameters())

        #print(' ')
        #return self.ckpt

if __name__ == '__main__':
    import argparse
    from parsers.config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpkt', type=int)

    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=bool, default=False)
    #parser.add_argument('--cpkt', type=int)

    args = parser.parse_args()
    

    cpkt = args.cpkt
    #for i in range(0,45):
    if True:
            #cpkt = 1 + 2 * i
        config = get_config(args.config, seed=42)
        trainer = Gnereatate3D(config)
        trainer.g(time.strftime('%b%d-%H:%M:%S', time.gmtime()),cpkt)
        
