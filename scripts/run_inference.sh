#!/bin/bash
#SBATCH --job-name=erdiff_inference
#SBATCH --output=slurm_output/inference_%x_%j.out
#SBATCH --error=slurm_output/inference_%x_%j.err
#SBATCH --partition="wu-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu="16G"

export PYTHONUNBUFFERED=TRUE

srun -u python3 -u inference.py \
    --model_path ../model_checkpoints/vae_model_mla.pth \
    --diffusion_model_path ../model_checkpoints/source_diffusion_model.pth \
    --source_data ../datasets/source_data_array.pkl \
    --target_data ../datasets/target_data_array.pkl \
    --output_dir ../outputs/ \
    --device cuda \
    --seed 2024
