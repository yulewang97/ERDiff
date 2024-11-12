import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm_notebook
import logging
import pickle
import scipy.signal as signal
from torch.utils.data import Dataset, DataLoader


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from model_functions.Diffusion import *
from model_functions.MLA_Model import *
from model_functions.VAE_Readout import *
from model_functions.ERDiff_utils import *
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import random

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

num_neurons_s, num_neurons_t = 187, 172

with open('../datasets-verify/source_data_arrays.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('../datasets-verify/target_data_arrays.pkl', 'rb') as f:
    test_data = pickle.load(f)


train_trial_spikes1, train_trial_vel1 = train_data['neural'], train_data['vel']

test_trial_spikes, test_trial_vel = test_data['neural'], test_data['vel']
# print(np.shape(train_trial_vel[0]))



bin_width = float(0.01) * 1000


train_trial_spikes_tide = train_trial_spikes1
train_trial_vel_tide = train_trial_vel1


kern_sd_ms = float(0.01) * 1000 * 5
kern_sd = int(round(kern_sd_ms / bin_width))
window = signal.gaussian(kern_sd, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

train_trial_spikes_smoothed = np.apply_along_axis(filt, 1, train_trial_spikes_tide)
test_trial_spikes_smoothed = np.apply_along_axis(filt, 1, test_trial_spikes)[:209]
test_trial_vel = test_trial_vel[:209]




timesteps = 40
eps = 1 / timesteps
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


input_dim = 1


diff_model = diff_STBlock(input_dim).to(device)

diff_model_dict = torch.load('../model_checkpoints/source_diffusion_model.pth')
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
setup_seed(2024)

vanilla_model_dict = torch.load('../model_checkpoints/source_vae_model.pth')

MLA_model = VAE_MLA_Model().to(device)
MLA_dict_keys = MLA_model.state_dict().keys()
vanilla_model_dict_keys = vanilla_model_dict.keys()

MLA_dict_new = MLA_model.state_dict().copy()

for key in vanilla_model_dict_keys:
    MLA_dict_new[key] = vanilla_model_dict[key]

MLA_model.load_state_dict(MLA_dict_new)


# Hyper
pre_total_loss_ = 1e8
l_rate = 3e-3
total_loss_list_ = []
last_improvement = 0
loss_list = []
key_metric = -1000
batch_size = 21
ot_weight = 0.8
appro_alpha = 0.05
epoches = 500




optimizer = torch.optim.SGD(MLA_model.parameters(), lr=l_rate)
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




test_trial_spikes_stand_half_len = len(test_trial_spikes_smoothed)


class SpikeDataset(Dataset):
    def __init__(self, spike_day_0, spike_day_k):
        self.spike_day_0 = spike_day_0
        self.spike_day_k = spike_day_k

    def __len__(self):
        return len(self.spike_day_0)

    def __getitem__(self, idx):
        return self.spike_day_0[idx], self.spike_day_k[idx]


spike_day_0 = Variable(torch.from_numpy(train_trial_spikes_smoothed)).float().to(device)
# spike_day_k = Variable(torch.from_numpy(test_trial_spikes_smoothed[:test_trial_spikes_stand_half_len])).float()
spike_day_k = Variable(torch.from_numpy(test_trial_spikes_smoothed)).float().to(device)
spike_dataset = SpikeDataset(spike_day_0, spike_day_k)

print(f'spike_day_0 shape: {spike_day_0.shape}, spike_day_k shape: {spike_day_k.shape}')

dataloader = DataLoader(spike_dataset, batch_size=batch_size, shuffle=False)

# 测试 DataLoader 的输出


num_x, num_y, num_y_test = spike_day_0.shape[0], spike_day_k.shape[0], test_trial_spikes_smoothed.shape[0]

p = Variable(torch.from_numpy(np.full((num_x, 1), 1 / num_x))).float().to(device)
q = Variable(torch.from_numpy(np.full((num_y, 1), 1 / num_y))).float().to(device)
q_test = Variable(torch.from_numpy(np.full((num_y_test, 1), 1 / num_y_test))).float().to(device)

def logger_performance(model):
    re_sp_test, vel_hat_test, _, _, _, _,_,_ = model(spike_day_0, spike_day_k, p, q_test, train_flag=False)

    sys.stdout.flush()
    y_true = test_trial_vel.reshape((-1, 2))
    y_pred = vel_hat_test.cpu().detach().numpy().reshape((-1, 2))


    key_metric = 100 * r2_score(y_true,y_pred, multioutput='uniform_average')
    return  key_metric

# Maximum Likelihood Alignment

for epoch in range(epoches):

    total_epoch_loss = 0
    total_ot_loss = 0
    total_diffusion_loss = 0 


    for batch in dataloader:
        optimizer.zero_grad()
        spike_day_0_batch, spike_day_k_batch = batch

        re_sp, _, distri_0, distri_k, latents_k, output_sh_loss, log_var, _ = MLA_model(spike_day_0, spike_day_k, p, q, train_flag=False) 

        ot_loss = ot_weight * output_sh_loss

        latents_k = latents_k[:, None, :, :]
        latents_k = torch.transpose(latents_k,3,2)

        batch_size = latents_k.shape[0]
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(latents_k, device=device)


        z_noisy = q_sample(x_start=latents_k, t=t, noise=noise)
        predicted_noise = diff_model(z_noisy, t)
        diffusion_loss = appro_alpha * F.smooth_l1_loss(noise, predicted_noise)
        

        total_loss = ot_loss + diffusion_loss

        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(MLA_model.parameters(), clip_value=2.0)
        

        optimizer.step()

        total_epoch_loss += total_loss.item()
        total_ot_loss += ot_loss.item()
        total_diffusion_loss += diffusion_loss.item()

    with torch.no_grad():
        print("Epoch: " + str(epoch) + " Total Loss: {:.4f}".format(total_epoch_loss))
        print("Epoch: " + str(epoch) + " OT Loss: {:.4f}".format(total_ot_loss))
        print("Epoch: " + str(epoch) + " Diffusion Loss: {:.4f}".format(total_diffusion_loss))
        current_metric = float(logger_performance(MLA_model))
        if current_metric > key_metric:
            key_metric = current_metric
        
        # if total_loss < pre_total_loss_:
        torch.save(MLA_model.state_dict(),'../model_checkpoints/vae_model_mla')
        pre_total_loss_ = total_loss 


        # Testing Phase
        _, _, _, _, test_latents, _,_, x_after_lowd = MLA_model(spike_day_0, spike_day_k,p,q_test, train_flag = False)
        test_latents = np.array(test_latents.cpu())

        vanilla_model_dict = torch.load('../model_checkpoints/vae_model_mla', weights_only=True)

        VAE_Readout_model = VAE_Readout_Model()
        DL_dict_keys = VAE_Readout_model.state_dict().keys()
        vanilla_model_dict_keys = vanilla_model_dict.keys()

        DL_dict_new = VAE_Readout_model.state_dict().copy()

        for key in vanilla_model_dict_keys:
            DL_dict_new[key] = vanilla_model_dict[key]

        VAE_Readout_model.load_state_dict(DL_dict_new)

        vel_cal(test_trial_vel, VAE_Readout_model, test_latents, x_after_lowd)
                    
