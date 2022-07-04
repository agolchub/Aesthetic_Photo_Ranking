#!/bin/sh
#SBATCH --partition=shortq7-gpu
#SBATCH  -N 1
#SBATCH --exclusive
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:1
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# modeule load shared tensorflow-gpu/2.6.0

# Execute the task
conda activate tf2-gpu
srun python ./train_segmented.py --special_model2 -o segment_complte.train -b 40 -e 100 -l 0.001 -d 0.0001 -t ./databaserelease2/train.csv -v ./databaserelease2/val.csv --width 244 --height 244 --outColumn 8 --build_only
