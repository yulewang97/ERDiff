import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import yaml
import sys
from datetime import datetime
from tqdm import tqdm_notebook
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import random
from torch.optim import Adam

import os
import sys


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import logging

from model_functions.diffusion import *
from model_functions.vae import *
from model_functions.erdiff_utils import *

logger = logging.getLogger('train_logger')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('train.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# logger.addHandler(console)
logger.info('python logging test')

import pickle


# Training``
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2024)



import numpy as np

timestamp = datetime.now().strftime("%m%d_%H%M")  
exp_name = f'Diffusion_Train_{timestamp}'


# define beta schedule
# betas = quadratic_beta_schedule(timesteps=timesteps)
betas = cosine_beta_schedule(timesteps=diff_num_steps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


channels = 1
global_batch_size = 32

# test_spike_data = test_spike_data.transpose(0,1,3,2)

from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Running the Train Code")

input_dim = 1


dm_model = diff_STBlock(input_dim)
dm_model.to(device)

ema = EMA(dm_model, decay=0.995)  


dm_optimizer = Adam(dm_model.parameters(), lr=1e-3)
# model

from torchvision.utils import save_image


pre_loss = 1e10


from torchvision import transforms
from torch.utils.data import DataLoader


import numpy as np
import scipy.interpolate


train_latents = np.load("../npy_files/train_latents.npy")
num_samples, seq_len, latent_dim = train_latents.shape

def add_gaussian_noise(latents, factor=0.08):
    noise_level = factor * np.std(latents)
    noisy_data = latents + np.random.normal(0, noise_level, latents.shape)
    return noisy_data

import numpy as np

def add_gaussian_noise(latents, factor=0.08):
    """Adds Gaussian noise to the latent representations."""
    noise_level = factor * np.std(latents)
    return latents + np.random.normal(0, noise_level, latents.shape)


noisy_latents = [add_gaussian_noise(train_latents) for _ in range(9)]


augmented_latents = np.concatenate([
    train_latents, 
    *noisy_latents 
], axis=0)



train_latents = augmented_latents



seq_len, latent_len = train_latents.shape[1], train_latents.shape[2]


train_latents = np.expand_dims(train_latents,1).astype(np.float32)
train_spike_data = train_latents.transpose(0,1,3,2)


dataloader = DataLoader(train_spike_data, batch_size=global_batch_size)

batch = next(iter(dataloader))

total_loss_array = np.zeros(n_epochs)

for epoch in range(n_epochs):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        dm_optimizer.zero_grad()

        batch_size = batch.shape[0]
        batch = batch.to(device)


        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(dm_model, batch, t)


        total_loss += loss.item()

        loss.backward()
        dm_optimizer.step()

        ema.update(dm_model)  
    
    total_loss_array[epoch] = total_loss


    print(f"total Loss of epoch {epoch} is {total_loss:.4f}")

    if total_loss < pre_loss:
        pre_loss = total_loss
        # torch.save(dm_model.state_dict(), f'../model_checkpoints/source_diffusion_model_public_{timesteps}.pth')
        torch.save(dm_model.state_dict(), f'../model_checkpoints/source_diffusion_model.pth')

# ema.apply(dm_model)
