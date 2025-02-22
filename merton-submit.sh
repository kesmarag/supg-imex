#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=12G
#SBATCH --time=1400 # time in minutes
#SBATCH --account=innovation

module load 2023r1 openmpi intel/oneapi-all py-torch py-pip py-matplotlib py-scipy


srun python merton_run_after.py

