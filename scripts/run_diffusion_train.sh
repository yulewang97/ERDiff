#!/bin/bash
#SBATCH --job-name=erdiff
#SBATCH --output=slurm_output/diffusion_train_%j.out
#SBATCH --error=slurm_output/diffusion_train_%j.err
#SBATCH --partition="wu-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu="32G"

export PYTHONUNBUFFERED=TRUE

srun -u python3 -u diffusion_train.py
