import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

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


len_trial,num_neurons = 37, 187

with open('datasets/Neural_Source.pkl', 'rb') as f:
    train_data1 = pickle.load(f)['data']


train_trial_spikes1, train_trial_vel1, train_trial_dir1 = train_data1['firing_rates'], train_data1['velocity'], train_data1['labels']

start_pos = 1 
end_pos = 1

train_trial_spikes_tide1 = np.array([spike[start_pos:len_trial+start_pos, :num_neurons] for spike in train_trial_spikes1])
print(np.shape(train_trial_spikes_tide1))

train_trial_vel_tide1 = np.array([spike[start_pos:len_trial+start_pos, :] for spike in train_trial_vel1])
print(np.shape(train_trial_vel_tide1))

# print(set(np.array(train_trial_dir)))
bin_width = float(0.02) * 1000


array_train_trial_dir1 = np.expand_dims(np.array((train_trial_dir1), dtype=object),1)

train_trial_spikes_tide = train_trial_spikes_tide1
train_trial_vel_tide = train_trial_vel_tide1
train_trial_dic_tide = np.squeeze(np.vstack([array_train_trial_dir1]))

import scipy.signal as signal

kern_sd_ms = 100
kern_sd = int(round(kern_sd_ms / bin_width))
window = signal.gaussian(kern_sd, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

train_trial_spikes_smoothed = np.apply_along_axis(filt, 1, train_trial_spikes_tide)

# test_trial_spikes_smoothed = test_trial_spikes_smoothed[:,1:,:]


indices = np.arange(train_trial_spikes_tide.shape[0])
np.random.seed(2023) 
np.random.shuffle(indices)
train_len = round(len(indices) * 0.80)
real_train_trial_spikes_smed, val_trial_spikes_smed = train_trial_spikes_smoothed[indices[:train_len]], train_trial_spikes_smoothed[indices[train_len:]]
real_train_trial_vel_tide, val_trial_vel_tide = train_trial_vel_tide[indices[:train_len]], train_trial_vel_tide[indices[train_len:]]
real_train_trial_dic_tide, val_trial_dic_tide = train_trial_dic_tide[indices[:train_len]], train_trial_dic_tide[indices[train_len:]]

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm_notebook

n_steps = 1
n_epochs = 500
batch_size = 16
ae_res_weight = 10
kld_weight = 1


from sklearn.metrics import r2_score


from sklearn.metrics import explained_variance_score
import random

n_batches = len(real_train_trial_spikes_smed)//batch_size
print(n_batches)
gamma_ = np.float32(1.)

mse_criterion = nn.MSELoss()
poisson_criterion = nn.PoissonNLLLoss(log_input=False)

l_rate = 0.001


real_train_trial_spikes_stand = (real_train_trial_spikes_smed)
val_trial_spikes_stand = (val_trial_spikes_smed)

spike_train = Variable(torch.from_numpy(real_train_trial_spikes_stand)).float()
spike_val = Variable(torch.from_numpy(val_trial_spikes_stand)).float()


emg_train = Variable(torch.from_numpy(real_train_trial_vel_tide)).float()
emg_val = Variable(torch.from_numpy(val_trial_vel_tide)).float()

def get_loss(model, spike, emg):
    re_sp_, vel_hat_,mu, log_var = model(spike, train_flag= True)
    ae_loss = poisson_criterion(re_sp_, spike)
    emg_loss = mse_criterion(vel_hat_, emg)
    kld_loss = torch.mean(0.5 * (- log_var + mu ** 2 + log_var.exp() - 1))
    total_loss = ae_res_weight * ae_loss + emg_loss + kld_weight * kld_loss
    # total_loss = ae_res_weight * ae_loss 
    return total_loss


# Training
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(21)


pre_total_loss_ = 1e18
total_loss_list_ = []
last_improvement = 0
loss_list = []


model = VAE_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

from torch.optim import Adam

import numpy as np

timesteps = 50

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

seq_len = 37
latent_len = 8

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
global_batch_size = 16

train_spikes = np.load("npy_files/train_latents.npy")


train_spike_data = np.expand_dims(train_spikes,1).astype(np.float32)
# test_spike_data = np.expand_dims(test_spikes,1).astype(np.float32)


train_spike_data = train_spike_data.transpose(0,1,3,2)
# test_spike_data = test_spike_data.transpose(0,1,3,2)



from torchvision import transforms
from torch.utils.data import DataLoader

dataloader = DataLoader(train_spike_data, batch_size=global_batch_size)

batch = next(iter(dataloader))
# print(batch.keys())

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


epochs = 500
pre_loss = 1e10


from torchvision import transforms
from torch.utils.data import DataLoader

for epoch in tqdm_notebook(range(n_epochs)):
    spike_gen_obj = get_batches(real_train_trial_spikes_stand,batch_size)
    emg_gen_obj = get_batches(real_train_trial_vel_tide,batch_size)
    for ii in range(n_batches):
        optimizer.zero_grad()
        spike_batch = next(spike_gen_obj)
        emg_batch = next(emg_gen_obj)

        spike_batch = Variable(torch.from_numpy(spike_batch)).float()
        emg_batch = Variable(torch.from_numpy(emg_batch)).float()

        # Loss
        batch_loss = get_loss(model, spike_batch, emg_batch)

        batch_loss.backward()
        optimizer.step()
        

    with torch.no_grad():
        val_total_loss = get_loss(model, spike_val, emg_val)
        loss_list.append(val_total_loss.item())

        _, _, train_latents, _ = model(spike_train, train_flag = False)

        if val_total_loss < pre_total_loss_: 
            pre_total_loss_ = val_total_loss
            torch.save(model.state_dict(),'model_checkpoints/source_vae_model')


            np.save("./npy_files/train_latents.npy",train_latents)

        
    train_latents = np.expand_dims(train_latents,1).astype(np.float32)
    train_spike_data = train_latents.transpose(0,1,3,2)


    dataloader = DataLoader(train_spike_data, batch_size=global_batch_size)

    batch = next(iter(dataloader))
    

    total_loss = 0
    for step, batch in enumerate(dataloader):
        dm_optimizer.zero_grad()

        batch_size = batch.shape[0]
        batch = batch.to(device)


        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(dm_model, batch, t)

        print("Step", step, " Loss:", loss.item())
        total_loss += loss.item()

        loss.backward()
        dm_optimizer.step()

    print("total Loss of epoch ", epoch, " is ", total_loss)

    if total_loss < pre_loss:
        pre_loss = total_loss
        torch.save(dm_model.state_dict(), 'model_checkpoints/source_diffusion_model')



