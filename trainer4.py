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
from loss import FlowMatchingLoss
from models.cnf2 import CNF


from utils.constants import num_to_letter


#from utils.loader2 import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
#                         load_ema, load_e3nn_loss_fn, load_ema_from_ckpt, \
#                         load_ckpt, load_opt_from_ckpt, load_model_from_ckpt


class Trainer3D(object):
    def __init__(self, config,ddp=False):
        super(Trainer3D, self).__init__()
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader, _, _ = get_dataloader(self.config,ddp=ddp)
        #self.params = load_model_params(self.config)

    def train(self, ts, resume=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        # -------- Load models, optimizers, ema --------
        self.model = CNF(GraphAttentionTransformer(**self.config.model).cuda())
        # self.model = torch.compile(self.model)
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr,
                                    weight_decay=self.config.train.weight_decay, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.train.lr_decay)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        #print(red(f'{self.ckpt}'))

        if resume and self.config.ckpt is not None:
            ckpt_dict = torch.load(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.config.ckpt}.pth')
            self.model.load_state_dict(ckpt_dict['state_dict'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.ema.load_state_dict(ckpt_dict['ema'])
            start_epoch = int(self.config.ckpt.split("_")[-1])
            print(f'Loaded checkpoint {self.config.ckpt}')
        else:
            start_epoch = 0

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)

        writer = SummaryWriter(os.path.join(*['logs_train', 'tensorboard', self.config.data.data,
                                            self.config.train.name, self.config.exp_name]))

        save_path = Path(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath('samples')
        sample_path.mkdir(exist_ok=True, parents=True)

        self.loss_fn = FlowMatchingLoss(self.model, self_cond=self.config.model.self_cond)

        train_pos_iter = []
        num_iter = 0
        write_to_tb = False # used to handle writing to tensorboard with gradient accumulation



        if True:
            if True:
                with torch.no_grad():
                    for batch in self.train_loader:
                        batch = batch.to(f'cuda:{self.device[0]}')
                        coords, mask, atom_mask, batch_id = batch.pos, batch.aa_mask, batch.atom_mask, batch.batch

                        aa = batch.aa_onehot

                        bb_coords = coords[:, :4]

                        #outPath = sample_path.joinpath(f'{self.ckpt}_{epoch+1}', f'epoch_{epoch+1}')
                        #outPath.mkdir(exist_ok=True, parents=True)

                        batch_size = batch_id.max().item() + 1

                        for r in range(self.config.sample.n_samples):
                            with torch.no_grad():
                                # Sampling
                                z = torch.randn(*bb_coords.shape).to(self.device[0])

                                z_aa = torch.randn_like(aa).to(self.device[0])

                                #z = torch.randn(bb_coords.shape[0],7+4,3).to(self.device[0])


                                pred_aa,pred_pos = self.model.decode(z_aa, z, batch)

                                #pred_aa = pred_all[:,:7,:]

                                #pred_aa = pred_aa.view(-1,21)

                                #pred_pos = pred_all[:,7:,:]
                                

                                print("all the shape here:",pred_aa.shape, pred_pos.shape)

                                pred_pos = pred_pos / self.config.data.scale_coords







        # -------- Training --------
        for epoch in trange(start_epoch, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):
            self.train_pos = []
            self.model.train()

            start_time = time.time()
            loss_sum = 0
            for idx, train_b in enumerate(self.train_loader):
                train_b = train_b.to('cuda')
                loss = self.loss_fn(train_b)
                loss = loss / config.train.grad_accum
                loss.backward()

                loss_sum += loss.item()
                if (idx + 1) % config.train.grad_accum == 0 or (idx + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.train_pos.append(loss_sum)
                    train_pos_iter.append(loss_sum)
                    loss_sum = 0
                    num_iter += 1
                    write_to_tb = True

                # -------- EMA update --------
                self.ema.update(self.model.parameters())

                if (num_iter+1) % 50 == 0 and write_to_tb:
                    writer.add_scalar("train_pos_iter", np.mean(train_pos_iter), num_iter+1)
                    train_pos_iter = []
                    write_to_tb = False

            if self.config.train.lr_schedule:
                self.scheduler.step()

            self.model.eval()
            test_loss_pos = []
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):
                    test_b = test_b.to('cuda')
                    loss = self.loss_fn(test_b)
                    test_loss_pos.append(loss.item())

            mean_test_pos = np.mean(test_loss_pos)
            mean_train_pos = np.mean(self.train_pos)

            writer.add_scalar("train_pos", mean_train_pos, epoch)
            writer.add_scalar("test_pos", mean_test_pos, epoch)
            writer.flush()

            # -------- Log losses --------
            logger.log(f'[EPOCH {epoch+1:04d}] | time: {time.time()-start_time:.2f} sec | '
                           f'train pos: {mean_train_pos:.3e} | test pos: {mean_test_pos:.3e}', verbose=False)

            # -------- Save checkpoints --------
            if epoch  % self.config.train.save_interval ==  0: #epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
                torch.save({ 
                    'config': self.config,
                    #'params' : self.params,
                    'state_dict': self.model.state_dict(), 
                    'ema': self.ema.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, save_path.joinpath(f'{save_name}.pth'))

            if  epoch  % self.config.train.save_interval ==  0: #epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] | time: {time.time()-start_time:.2f} sec | '
                           f'train pos: {mean_train_pos:.3e} | test pos: {mean_test_pos:.3e}')

            if  epoch  % self.config.train.save_interval ==  0: #epoch % self.config.train.sample_interval == self.config.train.sample_interval-1:
                # -------- Generate samples --------
                self.model.eval()
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model.parameters())

                with torch.no_grad():
                    for batch in self.train_loader:
                        batch = batch.to(f'cuda:{self.device[0]}')
                        coords, mask, atom_mask, batch_id = batch.pos, batch.aa_mask, batch.atom_mask, batch.batch
                        
                        aa = batch.aa_onehot

                        bb_coords = coords[:, :4]

                        outPath = sample_path.joinpath(f'{self.ckpt}_{epoch+1}', f'epoch_{epoch+1}')
                        outPath.mkdir(exist_ok=True, parents=True)

                        batch_size = batch_id.max().item() + 1

                        for r in range(self.config.sample.n_samples):
                            with torch.no_grad():
                                # Sampling
                                z = torch.randn(*bb_coords.shape).to(self.device[0])
                                
                                z_aa = torch.randn_like(aa).to(self.device[0])
 


                                pred_aa, pred_pos = self.model.decode(z_aa, z, batch)
                                pred_pos = pred_pos / self.config.data.scale_coords

                                # Save to PDB
                                for i in range(batch_size):
                                    pos_batch = pred_pos[batch_id == i]
                                    mask_batch = mask[batch_id == i]
                                    #aa = "G"*pos_batch.shape[0] # set to all glycines for now
                                    seq = pred_aa[batch_id == i]
                                    seq = torch.argmax(seq, dim=-1)
                                    m = torch.ones_like(seq)
                                    seq = ''.join([num_to_letter[j.item()] for h in seq])
                                    aa = seq

                                    pdb_path = outPath.joinpath(batch['id'][i])
                                    pdb_path.mkdir(exist_ok=True, parents=True)
                                    create_structure_from_crds(aa, pos_batch.cpu(), mask_batch.cpu(),
                                                               outPath=str(pdb_path.joinpath(f'run_{r + 1}.pdb')))
                        break
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
    trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()),resume=args.resume)
