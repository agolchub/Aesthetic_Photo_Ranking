#!/bin/sh
#SBATCH --partition=longq7-mri
#SBATCH  -N 1
#SBATCH --exclusive
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:1
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# modeule load shared tensorflow-gpu/2.6.0

# Execute the task
conda activate tf2-gpu
srun sh train.sh
