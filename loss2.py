import torch
import torch.nn as nn
from torch import Tensor
import random

class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module, eps=1e-5, self_cond=False):
        super().__init__()
        self.v = v
        self.eps = eps
        self.self_cond = self_cond

    def forward(self, batch) -> Tensor:
        aa, aa_num, edge_index, coords, mask, atom_mask, batch_id, pdb_id = batch.aa_onehot, batch.aa, batch.edge_index, \
                                                                            batch.pos, batch.aa_mask, batch.atom_mask, \
                                                                           batch.batch, batch.id
        

       # print("aa shape:", aa.shape)

       # print(aa)

        bb_crds = coords[:, :14]
        atom_mask = atom_mask[:,:14]

        batch_size = batch_id.max() + 1
        t = torch.rand([batch_size,1,1], device=aa.device)
        t = t[batch_id]
        z = torch.randn_like(bb_crds) * mask.view(-1, 1, 1) # i think here is mask reshape to -1,1,1 first then mutiple with torch.rand_like
        y = (1 - t) * bb_crds + (self.eps + (1 - self.eps) * t) * z
        u = (1 - self.eps) * z - bb_crds

        # the code below would be adding loss for aa.

        aa_z = torch.randn_like(aa)  * mask.view(-1, 1) # * mask.view(-1, 1, 1) # so aa shape should be B, length, 21

        # print("aa_z shape:", aa_z.shape)

        aa_y = (1 - t.squeeze(-1)) * aa + (self.eps + (1 - self.eps) * t.squeeze(-1)) * aa_z

        aa_u = (1 - self.eps) * aa_z - aa
        
        #print("aa_y shape:",aa_y.shape)

        # self-cond not implemented for sampling yet # I may need to update to include seq latter. 
        if self.self_cond:
            with torch.no_grad():
                pos_prev = torch.zeros_like(z)
                if random.random() > 0.5:
                    # First estimate
                    crds_in = torch.cat([y, pos_prev], dim=-2)
                    pos_prev = self.v(t.squeeze(-1), crds_in, edge_index, atom_mask, batch_id)
                y = torch.cat([y, pos_prev], dim=-2)

        pred_aa, pred_pos  = self.v(t.squeeze(-1),aa_y, y, edge_index, atom_mask, batch_id)
        loss_pos = (pred_pos - u).square().sum() / mask.sum()
        
        loss_aa = (pred_aa - aa_u).square().sum() / mask.sum()

        #print("loss_aa",loss_aa)
        #print("loss_pos",loss_pos)
        loss = loss_pos + loss_aa*0.5

        return loss
