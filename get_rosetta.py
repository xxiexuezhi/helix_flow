# calculate rosetta score from pdb

from pyrosetta.teaching import *

import pyrosetta
pyrosetta.init()


def rosetta_score(pdb_file_name):
    ras = pyrosetta.pose_from_pdb(pdb_file_name)
    sfxn = get_score_function(True)
    total_score = sfxn(ras)
    return total_score



# epoch 45
def sample_64_rosetta_lst(cpt):
    lst = []
    pre= "samples/epoch_"+str(cpt)+"/"
    for i in range(1,65):
        s = rosetta_score(pre+"run_"+str(i)+".pdb")
        lst.append(s)
    return lst



import sys

import statistics

cpt = int(sys.argv[1])
lst = sample_64_rosetta_lst(cpt)
print(lst, sum(lst)/len(lst),statistics.median(lst))
