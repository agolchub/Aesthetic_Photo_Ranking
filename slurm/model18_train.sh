#!/bin/sh
#SBATCH --partition=longq7-mri
#SBATCH  -N 1
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:1
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# modeule load shared tensorflow-gpu/2.6.0

# Execute the task
conda activate tf2-gpu
#srun python ./train_segmented.py --model_design 16 -o model16.3.train -b 5 --momentum 0.01 --loss mse -e 1000 -l 0.01 -d 0.01 --patience 50 -t ./databaserelease2/train2.csv -v ./databaserelease2/val2.csv --width 1024 --height 680 --outColumn 2 --reload_checkpoint_between_rates
srun python ./train_segmented.py --model_design 18 -o model18.2.train -b 5 --momentum 0.01 --loss mse -e 25 -l 0.01 -d 0.01 --patience 20 -t ./databaserelease2/train.csv -v ./databaserelease2/val.csv --width 1024 --height 680 --outColumn 16
