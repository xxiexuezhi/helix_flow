from Bio.PDB import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from utils.constants import num_to_letter, one_to_three_letter, restype_to_heavyatom_names

import torch

def create_structure_from_crds(aa,crds,mask,outPath="test.pdb", bb_only=False):

    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure(0)
    structure_builder.init_model(0)
    structure_builder.init_chain("A")
    structure_builder.init_seg(' ')

    for res_idx, res in enumerate(aa):
        aa_str = one_to_three_letter[res]
        if not mask[res_idx]: continue
        structure_builder.init_residue(aa_str," ",res_idx+1," ")

        if bb_only:
            atom_list = ['N','CA','C','O']
        else:
            atom_list = restype_to_heavyatom_names[one_to_three_letter[res]]

        for i,atom_name in enumerate(atom_list):
            if atom_name == '': continue
            if len(atom_name) == 1:
                fullname = f' {atom_name}  '
            elif len(atom_name) == 2:
                fullname = f' {atom_name} '
            elif len(atom_name) == 3:
                fullname = f' {atom_name}'
            else:
                fullname = atom_name  # len == 4
            structure_builder.init_atom(name=atom_name,coord=crds[res_idx,i],b_factor=res_idx+1.0,occupancy=1.0,altloc=" ",fullname=fullname)

    st = structure_builder.get_structure()
    io = PDBIO()
    io.set_structure(st)
    io.save(outPath)