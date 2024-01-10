#from mc_sc_to_residues import c_pdb
from utils.structure_utils import create_structure_from_crds

import pickle
import torch

name = "data/test_helix_data_encoded_using_coords_dec_2023_for_flow_matching_all.pkl"
name2 = "data/filter_rosetta100_helix_data_encoded_using_coords_dec_2023_for_flow_matching_all.pkl"

with open(name,"rb") as f:
    d1= pickle.load(f)

from dataset import ProteinDataset

train_ds = ProteinDataset(name)

#print(d1[66]['aa'],train_ds[66].aa_onehot)


import numpy as np
import math

from utils.constants import num_to_letter


from pyrosetta.teaching import *

import pyrosetta
pyrosetta.init()


def rosetta_score(pdb_file_name):
    ras = pyrosetta.pose_from_pdb(pdb_file_name)
    sfxn = get_score_function(True)
    total_score = sfxn(ras)
    return total_score





def get_sidechain_angles(sc_np):
    #s,m,sc_np = get_1dseq_main_side(arr)
    sc_angle_lst = []
    for i in range(len(sc_np)):
        angle_lst_one_res = []
        angle1 = np.arctan2(sc_np[i][0],sc_np[i][1])
        angle2 = np.arctan2(sc_np[i][2],sc_np[i][3])
        angle3 = np.arctan2(sc_np[i][4],sc_np[i][5])
        angle4 = np.arctan2(sc_np[i][6],sc_np[i][7])
        angle5 = np.arctan2(sc_np[i][8],sc_np[i][9])
        angle1=math.degrees(angle1)
        angle2=math.degrees(angle2)
        angle3=math.degrees(angle3)
        angle4=math.degrees(angle4)
        angle5=math.degrees(angle5)
        angle_lst_one_res.append(angle1)
        angle_lst_one_res.append(angle2)
        angle_lst_one_res.append(angle3)
        angle_lst_one_res.append(angle4)
        angle_lst_one_res.append(angle5)
        sc_angle_lst.append(angle_lst_one_res)
    return sc_angle_lst



def g_pdb_e3nn_with_rotamer(aa, mc, aa_sd,name):
    sd = aa_sd[:,21:].detach().numpy()
    sd_angle = get_sidechain_angles(sd)
    mc = mc[:,:3,:] # need to fix latter. this one is to reproduce the O atom posion
    mc = mc.reshape(-1,3)
    c_pdb(aa,mc,sd_angle,name)






print(train_ds[1].aa_onehot.shape)
def g_one_from_dataset(d_k):
    #aa = 
    aa = d_k.aa_onehot


    coords = d_k.pos
    #bb_coords = coords[:, :4]
    aa_mask = d_k.aa_mask
    seq = aa[:,:21]
    seq = torch.argmax(seq, dim=-1)
    #m = torch.ones_like(seq)
    seq = ''.join([num_to_letter[h.item()] for h in seq])


    #g_pdb_e3nn_with_rotamer(seq, bb_coords, aa,"tmp.pdb")
    create_structure_from_crds(seq, coords, aa_mask,outPath="tmp.pdb")

    score = rosetta_score("tmp.pdb")

    return score

s_lst = []
lst = []
c = 0
for i in range(200):#(len(d1)):
    s = g_one_from_dataset(train_ds[i])
    #print(s)
    s_lst.append(s)
    #lst.append(s)
    if s < 100:
        lst.append(d1[i])


print(s_lst)
# save as pickle file
#import pickle
#with open(name2,"wb") as fout:
#    pickle.dump(lst,fout)

#print(lst)
#print(c)
#print(rosetta_score("tmp.pdb"))
