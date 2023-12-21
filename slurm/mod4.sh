#!/bin/sh
#SBATCH --partition=longq7-mri
#SBATCH  -N 1
#SBATCH --exclusive
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:1
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# modeule load shared tensorflow-gpu/2.6.0

# Execute the task
srun python ./train_segmented.py --simple_model -o ./segmented_models/symmetry -b 40 -e 1000 -l 0.001 -d 0.0001 -t ./databaserelease2/train.csv -v ./databaserelease2/val.csv --width 244 --height 244 --outColumn 8
