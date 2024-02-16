#!/bin/bash

#SBATCH --job-name="Py_pi"
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=8G

module load 2022r2 openmpi py-torch
module load python/3.8.12
module load py-numpy
module load py-scipy
module load py-matplotlib


srun python case2.py > pi.log
