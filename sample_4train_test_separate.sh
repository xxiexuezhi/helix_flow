#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-83:05

module load cuda
#python train_heavy_light_chain.py 
#ipython train_heavy_light_chain2_ca_only.py
#python train_heavy_light_chain2.py

python large_sample_for_fullatom_helices.py sample_test_l4_aa_ab_seprate_aa_pos_in_cnf_morelayers --cpkt 45 
 
#python   trainer_seql4.py  test_l4_aa_ab  
