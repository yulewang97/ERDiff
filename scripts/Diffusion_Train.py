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

from model_functions.Diffusion import *
from model_functions.VAE import *
from model_functions.ERDiff_utils import *

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


# Hyper-parameters
l_rate = 1e-3
n_epochs = 400
timesteps = 40


# Training``
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2024)



import numpy as np



# define beta schedule
betas = quadratic_beta_schedule(timesteps=timesteps)

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

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


channels = 1
global_batch_size = 32

# test_spike_data = test_spike_data.transpose(0,1,3,2)



from torchvision import transforms
from torch.utils.data import DataLoader


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]

    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, seq_len, latent_len))

from pathlib import Path

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

dm_optimizer = Adam(dm_model.parameters(), lr=1e-3)
# model

from torchvision.utils import save_image


pre_loss = 1e10


from torchvision import transforms
from torch.utils.data import DataLoader




train_latents = np.load("../npy_files/train_latents.npy")


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

        # print("Step", step, " Loss:", loss.item())
        # print(f"total Loss of Step {step} is {loss.item():.4f}")

        total_loss += loss.item()

        loss.backward()
        dm_optimizer.step()
    
    total_loss_array[epoch] = total_loss
    print(f"total Loss of epoch {epoch} is {total_loss:.4f}")

    if total_loss < pre_loss:
        pre_loss = total_loss
        torch.save(dm_model.state_dict(), '../model_checkpoints/source_diffusion_model.pth')

np.save('../npy_files/diffusion_total_loss_array.npy',total_loss_array)