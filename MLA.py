import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm_notebook
import logging
import pickle
import scipy.signal as signal
from model_functions.Diffusion import *
from model_functions.MLA_Model import *
from model_functions.VAE_Readout import *
from model_functions.ERDiff_utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm_notebook

logger = logging.getLogger('train_logger')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('train.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# logger.addHandler(console)
logger.info('python logging test')

len_trial, num_neurons_s, num_neurons_t = 37, 187, 172

with open('datasets/Neural_Source.pkl', 'rb') as f:
    train_data1 = pickle.load(f)['data']

with open('datasets/Neural_Target.pkl', 'rb') as f:
    test_data = pickle.load(f)['data']


train_trial_spikes1, train_trial_vel1, train_trial_dir1 = train_data1['firing_rates'], train_data1['velocity'], train_data1['labels']

test_trial_spikes, test_trial_vel, test_trial_dir = test_data['firing_rates'], test_data['velocity'], np.squeeze(test_data['labels'])
# print(np.shape(train_trial_vel[0]))
start_pos = 1

train_trial_spikes_tide1 = np.array([spike[start_pos:len_trial+start_pos, :num_neurons_s] for spike in train_trial_spikes1])
print(np.shape(train_trial_spikes_tide1))

train_trial_vel_tide1 = np.array([spike[start_pos:len_trial+start_pos, :] for spike in train_trial_vel1])
print(np.shape(train_trial_vel_tide1))


test_trial_spikes_tide = np.array([spike[:len_trial, :num_neurons_t] for spike in test_trial_spikes])
print(np.shape(test_trial_spikes_tide))


test_trial_vel_tide = np.array([spike[:len_trial, :] for spike in test_trial_vel])
print(np.shape(test_trial_vel_tide))

bin_width = float(0.02) * 1000

array_train_trial_dir1 = np.expand_dims(np.array((train_trial_dir1)),1)

train_trial_spikes_tide = train_trial_spikes_tide1
train_trial_vel_tide = train_trial_vel_tide1
train_trial_dic_tide = np.squeeze(np.vstack([array_train_trial_dir1]))
test_trial_dic_tide = np.squeeze(np.vstack([test_trial_dir]))

kern_sd_ms = 100
kern_sd = int(round(kern_sd_ms / bin_width))
window = signal.gaussian(kern_sd, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

train_trial_spikes_smoothed = np.apply_along_axis(filt, 1, train_trial_spikes_tide)
test_trial_spikes_smoothed = np.apply_along_axis(filt, 1, test_trial_spikes_tide)


indices = np.arange(train_trial_spikes_tide.shape[0])
np.random.seed(2023) 
np.random.shuffle(indices)
train_len = round(len(indices) * 0.8)
real_train_trial_spikes_smed, val_trial_spikes_smed = train_trial_spikes_smoothed[indices[:train_len]], train_trial_spikes_smoothed[indices[train_len:]]
real_train_trial_vel_tide, val_trial_vel_tide = train_trial_vel_tide[indices[:train_len]], train_trial_vel_tide[indices[train_len:]]
real_train_trial_dic_tide, val_trial_dic_tide = train_trial_dic_tide[indices[:train_len]], train_trial_dic_tide[indices[train_len:]]


n_steps = 1
n_epochs = 500
batch_size = 64


from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import random


l_rate = 0.001

real_train_trial_spikes_stand = (real_train_trial_spikes_smed)
val_trial_spikes_stand = (val_trial_spikes_smed)
test_trial_spikes_stand = (test_trial_spikes_smoothed)

spike_train = Variable(torch.from_numpy(real_train_trial_spikes_stand)).float()
spike_val = Variable(torch.from_numpy(val_trial_spikes_stand)).float()
spike_test = Variable(torch.from_numpy(test_trial_spikes_stand)).float()


timesteps = 50
eps = 1 / timesteps
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


input_dim = 1


diff_model = diff_STBlock(input_dim)

diff_model_dict = torch.load('model_checkpoints/source_diffusion_model')
diff_model.load_state_dict(diff_model_dict)

for k,v in diff_model.named_parameters():
    v.requires_grad=False


import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(21)

vanilla_model_dict = torch.load('model_checkpoints/source_vae_model')

MLA_model = VAE_MLA_Model()
MLA_dict_keys = MLA_model.state_dict().keys()
vanilla_model_dict_keys = vanilla_model_dict.keys()

MLA_dict_new = MLA_model.state_dict().copy()

for key in vanilla_model_dict_keys:
    MLA_dict_new[key] = vanilla_model_dict[key]

MLA_model.load_state_dict(MLA_dict_new)


pre_total_loss_ = 1e18
l_rate = 1e-3
total_loss_list_ = []
last_improvement = 0
loss_list = []
key_metric = -1000
appro_alpha = 1.2

optimizer = torch.optim.Adam(MLA_model.parameters(), lr=l_rate)
criterion = nn.MSELoss()
poisson_criterion = nn.PoissonNLLLoss(log_input=False)

# for param in MLA_model.parameters():
#     param.requires_grad = False

for param in MLA_model.vde_rnn.parameters():
    param.requires_grad = False

for param in MLA_model.sde_rnn.parameters():
    param.requires_grad = False
    
for param in MLA_model.encoder_rnn.parameters():
    param.requires_grad = False

MLA_model.low_d_readin_s.weight.requires_grad = False
MLA_model.low_d_readin_s.bias.requires_grad = False
MLA_model.fc_mu_1.weight.requires_grad = False
MLA_model.fc_mu_1.bias.requires_grad = False
MLA_model.fc_log_var_1.weight.requires_grad = False
MLA_model.fc_log_var_1.bias.requires_grad = False
MLA_model.sde_fc1.weight.requires_grad = False
MLA_model.sde_fc1.bias.requires_grad = False
MLA_model.sde_fc2.weight.requires_grad = False
MLA_model.sde_fc2.bias.requires_grad = False
MLA_model.vde_fc_minus_0.weight.requires_grad = False


epoches = 300
test_trial_spikes_stand_half_len = len(test_trial_spikes_stand) // 2


spike_day_0 = Variable(torch.from_numpy(real_train_trial_spikes_stand)).float()
spike_day_k = Variable(torch.from_numpy(test_trial_spikes_stand[:test_trial_spikes_stand_half_len])).float()


num_x, num_y, num_y_test = spike_day_0.shape[0], spike_day_k.shape[0], test_trial_spikes_stand.shape[0]

p = Variable(torch.from_numpy(np.full((num_x, 1), 1 / num_x))).float()
q = Variable(torch.from_numpy(np.full((num_y, 1), 1 / num_y))).float()
q_test = Variable(torch.from_numpy(np.full((num_y_test, 1), 1 / num_y_test))).float()

def logger_performance(model):
    re_sp_test, vel_hat_test, _, _, _, _,_ = model(spike_train, spike_test, p, q_test, train_flag=False)

    sys.stdout.flush()
    key_metric = 100 * r2_score(test_trial_vel_tide.reshape((-1,2)),vel_hat_test.reshape((-1,2)), multioutput='uniform_average')
    return  key_metric

# Maximum Likelihood Alignment

for epoch in range(epoches):

    optimizer.zero_grad()

    re_sp, _, distri_0, distri_k, latents_k, output_sh_loss, log_var = MLA_model(spike_day_0, spike_day_k, p, q, train_flag=True) 

    total_loss = output_sh_loss

    latents_k = latents_k[:, None, :, :]
    latents_k = torch.transpose(latents_k,3,2)

    batch_size = latents_k.shape[0]
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(latents_k)


    z_noisy = q_sample(x_start=latents_k, t=t, noise=noise)
    predicted_noise = diff_model(z_noisy, t)
    total_loss += appro_alpha * F.smooth_l1_loss(noise, predicted_noise)
    
    total_loss += skilling_divergence(z_noisy,latents_k,t)
    

    total_loss.backward(retain_graph=True)
    optimizer.step()

    with torch.no_grad():
        if (epoch % 5 == 0) or (epoch == n_epochs-1):
            print(total_loss)
            logger.info("Epoch:" + str(epoch) )
            current_metric = float(logger_performance(MLA_model))
            if current_metric > key_metric:
                key_metric = current_metric
            if total_loss < pre_total_loss_:
                torch.save(MLA_model.state_dict(),'model_checkpoints/vae_model_mla')
                pre_total_loss_ = total_loss 
                

vanilla_model_dict = torch.load('model_checkpoints/vae_model_mla')

MLA_model = VAE_MLA_Model()

MLA_model.load_state_dict(vanilla_model_dict)

with torch.no_grad():
    _, _, _, _, test_latents, _,_ = MLA_model(spike_train, spike_test,p,q_test, train_flag = False)
test_latents = np.array(test_latents)
np.save("./npy_files/test_latents.npy",test_latents)
    

def create_dir_dict(trial_dir):
    dir_dict = {}
    for i, dir in enumerate(trial_dir):
        dir = dir[0][0]
        if not np.isnan(dir):
            if dir not in dir_dict:
                dir_dict[dir] = [i]
            else:
                dir_dict[dir].append(i)
    return dir_dict

train_dir_dict, test_dir_dict = create_dir_dict(real_train_trial_dic_tide), create_dir_dict(test_trial_dir)
val_dir_dict = create_dir_dict(val_trial_dic_tide)


vanilla_model_dict = torch.load('model_checkpoints/vae_model_mla')

VAE_Readout_model = VAE_Readout_Model()
DL_dict_keys = VAE_Readout_model.state_dict().keys()
vanilla_model_dict_keys = vanilla_model_dict.keys()

DL_dict_new = VAE_Readout_model.state_dict().copy()

for key in vanilla_model_dict_keys:
    DL_dict_new[key] = vanilla_model_dict[key]

VAE_Readout_model.load_state_dict(DL_dict_new)

vel_cal(test_trial_vel_tide, VAE_Readout_model, test_latents)