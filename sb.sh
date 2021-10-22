#!/bin/sh
#SBATCH --partition=shortq7-gpu
#SBATCH  -N 1
#SBATCH --exclusive
#SBATCH --mem-per-cpu=16000
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# Execute the task

srun sh train.sh
