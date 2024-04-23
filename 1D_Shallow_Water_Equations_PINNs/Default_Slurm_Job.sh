#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=default_settings
#SBATCH --nodes=1

## The number of tasks per node should be the same number as requested GPUS per node.
#SBATCH --ntasks-per-node=1

## The number of cpus per task should be the same number as dataloader workers.
#SBATCH --cpus-per-task=1

#SBATCH --time=3-00:00:00
#SBATCH --account=demir
#SBATCH --partition=pGPU
#SBATCH --exclusive

## output directory should already exist
#SBATCH --output=slurm_output/default_settings/slurm-%j.out

mkdir $SLURM_JOB_ID
module load compilers/cuda/11.0
nvidia-smi
srun /gpfs/home/demir/miniconda3/envs/plankton/bin/python /gpfs/home/demir/Testing-PINNs/1D_Shallow_Water_Equations/Train_SWE.py
