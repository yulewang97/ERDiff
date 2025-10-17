"""
ERDiff Inference Script

This script performs inference using the trained VAE-MLA and diffusion models
to decode neural activity and predict velocities for the target session.
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import scipy.signal as signal
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Argument parser for command line options
parser = argparse.ArgumentParser(description="ERDiff Inference Script")

parser.add_argument("--model_path", type=str, default='../model_checkpoints/vae_model_mla.pth', 
                    help="Path to the trained MLA model")
parser.add_argument("--diffusion_model_path", type=str, default='../model_checkpoints/source_diffusion_model.pth',
                    help="Path to the trained diffusion model")
parser.add_argument("--source_data", type=str, default='../datasets/source_data_array.pkl',
                    help="Path to source data pickle file")
parser.add_argument("--target_data", type=str, default='../datasets/target_data_array.pkl',
                    help="Path to target data pickle file")
parser.add_argument("--output_dir", type=str, default='../outputs/',
                    help="Directory to save inference results")
parser.add_argument("--device", type=str, default='cuda',
                    help="Device to run inference on (cuda or cpu)")
parser.add_argument("--seed", type=int, default=2024, 
                    help="Random seed for reproducibility")

args = parser.parse_args()

# Import model functions
from model_functions.diffusion import diff_STBlock, q_sample
from model_functions.mla_model import VAE_MLA_Model
from model_functions.vae_readout import VAE_Readout_Model
from utils_scripts.utils_torch import vel_cal, create_dir_dict, setup_seed

# Set random seed for reproducibility
setup_seed(args.seed)

# Device configuration
device = args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu"
print(f"Using device: {device}")

# Load data
print("Loading data...")
with open(args.source_data, 'rb') as f:
    train_data = pickle.load(f)

with open(args.target_data, 'rb') as f:
    test_data = pickle.load(f)

train_trial_spikes, train_trial_vel, train_trial_dir = train_data['neural'], train_data['vel'], train_data['label']
test_trial_spikes, test_trial_vel, test_trial_dir = test_data['neural'], test_data['vel'], np.squeeze(test_data['label'])

print(f"Source data shape: {train_trial_spikes.shape}")
print(f"Target data shape: {test_trial_spikes.shape}")

# Smooth the spike data
bin_width = float(0.01) * 1000
kern_sd_ms = float(0.01) * 1000 * 3
kern_sd = int(round(kern_sd_ms / bin_width))
window = signal.gaussian(kern_sd, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

train_trial_spikes_smoothed = np.apply_along_axis(filt, 1, train_trial_spikes)
test_trial_spikes_smoothed = np.apply_along_axis(filt, 1, test_trial_spikes)

# Convert to tensors
spike_day_0 = Variable(torch.from_numpy(train_trial_spikes_smoothed)).float().to(device)
spike_day_k = Variable(torch.from_numpy(test_trial_spikes_smoothed)).float().to(device)

num_x, num_y_test = spike_day_0.shape[0], test_trial_spikes_smoothed.shape[0]
p = Variable(torch.from_numpy(np.full((num_x, 1), 1 / num_x))).float().to(device)
q_test = Variable(torch.from_numpy(np.full((num_y_test, 1), 1 / num_y_test))).float().to(device)

# Load models
print("Loading models...")

# Load MLA model
MLA_model = VAE_MLA_Model().to(device)
MLA_model_dict = torch.load(args.model_path, map_location=torch.device(device), weights_only=True)
MLA_model.load_state_dict(MLA_model_dict)
MLA_model.eval()

# Load Readout model (using same weights as MLA)
VAE_Readout_model = VAE_Readout_Model()
DL_dict_keys = VAE_Readout_model.state_dict().keys()
DL_dict_new = VAE_Readout_model.state_dict().copy()

for key in MLA_model_dict.keys():
    if key in DL_dict_new:
        DL_dict_new[key] = MLA_model_dict[key]

VAE_Readout_model.load_state_dict(DL_dict_new)
VAE_Readout_model.eval()

print("Models loaded successfully!")

# Run inference
print("Running inference...")
with torch.no_grad():
    _, _, _, _, test_latents, _, _, x_after_lowd = MLA_model(
        spike_day_0, spike_day_k, p, q_test, train_flag=False
    )
    test_latents = test_latents.cpu().numpy()

# Calculate velocity predictions
print("\nVelocity prediction results:")
vel_cal(test_trial_vel, VAE_Readout_model, torch.Tensor(test_latents), x_after_lowd)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(args.output_dir, f'inference_results_{timestamp}.npz')

np.savez(
    output_file,
    test_latents=test_latents,
    test_trial_vel=test_trial_vel,
    x_after_lowd=x_after_lowd.cpu().numpy() if isinstance(x_after_lowd, torch.Tensor) else x_after_lowd
)

print(f"\nResults saved to: {output_file}")
print("Inference complete!")
