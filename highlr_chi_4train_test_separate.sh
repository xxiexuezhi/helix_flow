#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --gres=gpu:v100:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-99:05

module load cuda
#python train_heavy_light_chain.py 
#ipython train_heavy_light_chain2_ca_only.py
#python train_heavy_light_chain2.py

python trainer_seql4_seprate_aa_pos_in_cnf.py test_l4_aa_ab_seprate_aa_pos_in_cnf_highlr --resume true
 
#python   trainer_seql4.py  test_l4_aa_ab  
