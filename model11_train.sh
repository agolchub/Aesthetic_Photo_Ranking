#!/bin/sh
#SBATCH --partition=shortq7-gpu
#SBATCH  -N 1
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:1
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# modeule load shared tensorflow-gpu/2.6.0

# Execute the task
conda activate tf2-gpu
srun python ./train_segmented.py --model_design 11 -o model11.1.train -b 5 --loss bce -e 50 -l 0.1 -t ./databaserelease2/train.csv -v ./databaserelease2/val.csv --width 1024 --height 1024 --outColumn 10
srun python ./train_segmented.py --model_design 11 -o model11.1.train -b 5 --loss bce -e 100 -l 0.01 -t ./databaserelease2/train.csv -v ./databaserelease2/val.csv --width 1024 --height 1024 --outColumn 10
srun python ./train_segmented.py --model_design 11 -o model11.1.train -b 5 --loss bce -e 100 -l 0.001 -t ./databaserelease2/train.csv -v ./databaserelease2/val.csv --width 1024 --height 1024 --outColumn 10