#!/bin/sh
#SBATCH – – partition=shortq7
#SBATCH  -N 1
#SBATCH – -exclusive
#SBATCH – – mem-per-cpu=16000
# Load modules, if needed, run staging tasks, etc… Koko Software Modules
# Execute the task

srun python ./train.py --special_model -o model2.train -b 5 -e 2 -r -d 0.0001 -l 0.001 -t ./databaserelease2/NatureDataset/train/ -v ./databaserelease2/NatureDataset/val/ --build_only