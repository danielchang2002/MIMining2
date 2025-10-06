#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --partition=preemptible,cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512GB

# python3 DFT_analysis.py
python3 DFT_analysis2.py
