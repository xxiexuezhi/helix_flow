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



lst = []
import sys

import statistics
for i in range (71,86):#(60,71):#(40,60):#(139,164): #(99,135):#(23,39):
    #for cpkt in [77,79,81]:
        #if True:
    cpt = 1 + 2 * i
#cpt = int(sys.argv[1])
    #sub_lst = sample_64_rosetta_lst(cpt)
    lst.append(sample_64_rosetta_lst(cpt))

#print(lst, sum(lst)/len(lst),statistics.median(lst))

print(lst)
