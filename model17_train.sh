#!/bin/sh
#SBATCH --partition=shortq7-gpu
#SBATCH  -N 1
#SBATCH --mem-per-cpu=64000
#SBATCH --gres=gpu:1
# Load modules, if needed, run staging tasks, etc... Koko Software Modules
# modeule load shared tensorflow-gpu/2.6.0

# Execute the task
conda activate tf2-gpu
srun python ./train_segmented.py --model_design 17 -o model17.1.train -b 5 --momentum 0.01 --loss mse -e 1000 -l 0.01 -d 0.01 --patience 50 -t ./databaserelease2/train2.csv -v ./databaserelease2/val2.csv --width 1024 --height 680 --outColumn 2 --reload_checkpoint_between_rates
