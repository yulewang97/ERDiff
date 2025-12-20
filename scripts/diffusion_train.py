from inspect import isfunction

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import os
import wandb
import sys
from datetime import datetime
from tqdm import tqdm_notebook
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import random
from torch.optim import Adam
from torchvision.utils import save_image

from torchvision import transforms
from torch.utils.data import DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import logging

from model_functions.diffusion import *
from model_functions.vae import *
from model_functions.dataloader import AddGaussianNoise, LatentDataset
from torchvision.transforms import ToTensor

logger = logging.getLogger('train_logger')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('train.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info('python logging test')


# Training
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2024)


timestamp = datetime.now().strftime("%m%d_%H%M")
exp_name = f'Diffusion_Train{timestamp}'
wandb.init(project="ERDiff", name=exp_name, config={})

# define beta schedule
# betas = quadratic_beta_schedule(timesteps=timesteps)
betas = cosine_beta_schedule(diff_timesteps=diff_timesteps)

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


channels = 1
batch_size = 32


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 1

dm_model = diff_STBlock(input_dim)
dm_model.to(device)

ema = EMA(dm_model, decay=0.995)


dm_optimizer = Adam(dm_model.parameters(), lr=1e-3)
# model

pre_loss = float('inf')

# Data Loading
train_latents = np.load("../npy_files/train_latents.npy")
num_samples, seq_len, latent_dim = train_latents.shape

transform = transforms.Compose([
    ToTensor(),
    AddGaussianNoise(factor=0.075),
])

dataset = LatentDataset(train_latents, transform=transform)

logger.info(f"Data Shape after Augmentation: {train_latents.shape}")

seq_len, latent_len = train_latents.shape[1], train_latents.shape[2]

train_latents = np.expand_dims(train_latents,1).astype(np.float32)
train_spike_data = train_latents.transpose(0,1,3,2)

dataloader = DataLoader(dataset, batch_size=batch_size)

batch = next(iter(dataloader))

total_loss_array = np.zeros(n_epochs)

for epoch in range(n_epochs):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        dm_optimizer.zero_grad()

        batch_size = batch.shape[0]
        batch = batch.to(device)

        t = torch.randint(0, diff_timesteps, (batch_size,), device=device).long()

        loss = p_losses(dm_model, batch, t)

        total_loss += loss.item()

        loss.backward()
        dm_optimizer.step()

        ema.update(dm_model)  # update EMA parameters
    
    total_loss_array[epoch] = total_loss
    wandb.log({
        "total_epoch_loss": total_loss / 1.
    })

    logger.info(f"total Loss of epoch {epoch} is {total_loss:.4f}")

    if total_loss < pre_loss:
        pre_loss = total_loss
        torch.save(dm_model.state_dict(), f'../model_checkpoints/source_diffusion_model.pth')
