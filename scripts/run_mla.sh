#!/bin/bash
#SBATCH --job-name=erdiff
#SBATCH --output=slurm_output/mla_%x_%j.out
#SBATCH --error=slurm_output/mla_%x_%j.err
#SBATCH --partition="wu-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="long"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu="32G"

export PYTHONUNBUFFERED=TRUE

srun -u python3 -u mla.py \
    --learning_rate 3e-3 \
    --batch_size 64 \
    --appro_alpha 0.10 \
    --ot_weight 1.0 \
    --epoches 1000 \
    --seed 2024
