#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --gres=gpu:v100:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-23:05

module load cuda
#python train_heavy_light_chain.py 
#ipython train_heavy_light_chain2_ca_only.py
#python train_heavy_light_chain2.py

python  trainer_seq2.py test_l3_aa --resume true
 
