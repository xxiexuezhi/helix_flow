import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Normal
from zuko.utils import odeint

class CNF(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z_aa: Tensor, x: Tensor, batch) -> Tensor:

        edge_index, atom_mask, batch_id = batch.edge_index, batch.atom_mask, batch.batch
        atom_mask = atom_mask[:,:14]


        # z_all is the merged tensor where frist 7,3 are aa and rest 4,3 are pos.
        # I tried to separte the two varaible aa and pos like https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa

        def func(t: Tensor, z_aa: Tensor, x: Tensor) -> Tensor:
            
            #print("z_all", z_all.shape)

            #z_aa = z_all[:,:7,:]
            
            ##z_aa = z_aa.view([-1,21])
            
            #x = z_all[:,7:,:]

            pred_aa,pred_pos = self(t, z_aa,  x, edge_index, atom_mask, batch_id)
            #print("pred_aa", pred_aa.shape)
            #print("pred_pos",pred_pos.shape)
            #pred_aa_reshape = pred_aa.view(-1,7,3)
            return pred_aa, pred_pos #torch.cat((pred_aa_reshape, pred_pos), dim=1)

            
        #func = lambda t,x: self(t, *z_aa,  x, edge_index, atom_mask, batch_id)

        return odeint(func, (z_aa, x), 1.0, 0.0, phi=self.parameters())

#    def decode(self, z: Tensor, batch) -> Tensor:
#        edge_index, atom_mask, batch_id = batch.edge_index, batch.atom_mask, batch.batch
#        atom_mask = atom_mask[:,:4]
#        func = lambda t,x: self(t, x, edge_index, atom_mask, batch_id)
#        return odeint(func, z, 1.0, 0.0, phi=self.parameters())



    # Hutchinson trace estimation
    def log_prob(self, x: Tensor) -> Tensor:
        eps = torch.randn_like(x)
        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, eps)[0]
            trace = torch.sum(jacobian * eps, dim=tuple(range(1, len(x.shape))))

            return dx, trace

        ladj = torch.zeros_like(x[...,0]) # just placeholder for likelihood
        z, ladj = odeint(augmented, (x,ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj

