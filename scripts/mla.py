import scipy.io as sio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook
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



parser = argparse.ArgumentParser(description="Set hyperparameters from command line")


parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--appro_alpha", type=float, default=0.0, help="Approximator alpha parameter")
parser.add_argument("--ot_weight", type=float, default=0.8, help="Weight for optimal transport loss")
parser.add_argument("--epoches", type=int, default=400, help="Alternative epoch count (possible typo in config)")
parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")


args = parser.parse_args()


config = vars(args)


print("Config Data:", config)


from model_functions.diffusion import diff_STBlock, q_sample
from model_functions.mla_model import VAE_MLA_Model
from model_functions.vae_readout import VAE_Readout_Model
from utils.utils_torch import vel_cal, create_dir_dict
from utils.utils_torch import setup_seed, SpikeDataset, logger_performance
from torch.utils.data import DataLoader
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score




num_neurons_s, num_neurons_t = 187, 172

with open('../datasets/source_data_array.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('../datasets/target_data_array.pkl', 'rb') as f:
    test_data = pickle.load(f)


train_trial_spikes1, train_trial_vel1, train_trial_dir1 = train_data['neural'], train_data['vel'], train_data['label']

test_trial_spikes, test_trial_vel, test_trial_dir = test_data['neural'], test_data['vel'], np.squeeze(test_data['label'])
# print(np.shape(train_trial_vel[0]))



bin_width = float(0.01) * 1000

array_train_trial_dir1 = np.expand_dims(np.array(train_trial_dir1, dtype=object),1)

train_trial_spikes_tide = train_trial_spikes1
train_trial_vel_tide = train_trial_vel1
train_trial_dic_tide = np.squeeze(np.vstack([array_train_trial_dir1]))
test_trial_dic_tide = np.squeeze(np.vstack([test_trial_dir]))

kern_sd_ms = float(0.01) * 1000 * 3
kern_sd = int(round(kern_sd_ms / bin_width))
window = signal.gaussian(kern_sd, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

train_trial_spikes_smoothed = np.apply_along_axis(filt, 1, train_trial_spikes_tide)
test_trial_spikes_smoothed = np.apply_along_axis(filt, 1, test_trial_spikes)
test_trial_vel = test_trial_vel


timesteps = 100
eps = 1 / timesteps
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"device: {device}")


input_dim = 1


diff_model = diff_STBlock(input_dim).to(device)

diff_model_dict = torch.load('../model_checkpoints/source_diffusion_model.pth', map_location=torch.device('cpu'), weights_only=True)
diff_model.load_state_dict(diff_model_dict)

for k,v in diff_model.named_parameters():
    v.requires_grad=False



setup_seed(config["seed"])

vanilla_model_dict = torch.load('../model_checkpoints/source_vae_model.pth', weights_only=True, map_location=torch.device('cpu'))

MLA_model = VAE_MLA_Model().to(device)
MLA_dict_keys = MLA_model.state_dict().keys()
vanilla_model_dict_keys = vanilla_model_dict.keys()

MLA_dict_new = MLA_model.state_dict().copy()

for key in vanilla_model_dict_keys:
    MLA_dict_new[key] = vanilla_model_dict[key]

MLA_model.load_state_dict(MLA_dict_new)


pre_total_loss_ = 1e8
best_metric = -1000
l_rate = config["learning_rate"]
batch_size = config["batch_size"]
ot_weight = config["ot_weight"]
appro_alpha = config["appro_alpha"]



optimizer = torch.optim.SGD(MLA_model.parameters(), lr=l_rate)
criterion = nn.MSELoss()
poisson_criterion = nn.PoissonNLLLoss(log_input=False)

# Freeze the other parameters

for param in MLA_model.parameters():
    param.requires_grad = False


MLA_model.low_d_readin_t_2.weight.requires_grad = True
MLA_model.low_d_readin_t_2.bias.requires_grad = True


epoches = config["epoches"]
# test_trial_spikes_stand_half_len = len(test_trial_spikes_smoothed) // 2
test_trial_spikes_stand_half_len = len(test_trial_spikes_smoothed)




spike_day_0 = Variable(torch.from_numpy(train_trial_spikes_smoothed)).float().to(device)
# spike_day_k = Variable(torch.from_numpy(test_trial_spikes_smoothed[:test_trial_spikes_stand_half_len])).float()
spike_day_k = Variable(torch.from_numpy(test_trial_spikes_smoothed)).float().to(device)
spike_dataset = SpikeDataset(spike_day_0, spike_day_k)

print(f'spike_day_0 shape: {spike_day_0.shape}, spike_day_k shape: {spike_day_k.shape}')

dataloader = DataLoader(spike_dataset, batch_size=batch_size, shuffle=False)




num_x, num_y, num_y_test = spike_day_0.shape[0], spike_day_k.shape[0], test_trial_spikes_smoothed.shape[0]

p = Variable(torch.from_numpy(np.full((num_x, 1), 1 / num_x))).float().to(device)
q = Variable(torch.from_numpy(np.full((num_y, 1), 1 / num_y))).float().to(device)
q_test = Variable(torch.from_numpy(np.full((num_y_test, 1), 1 / num_y_test))).float().to(device)

timestamp = datetime.now().strftime("%m%d_%H%M")  
exp_name = f'ERDiff_MLA_{timestamp}'



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
        diffusion_loss = appro_alpha * F.mse_loss(noise, predicted_noise)
        

        total_loss = ot_loss + diffusion_loss

        total_loss.backward(retain_graph=True)

        optimizer.step()

        total_epoch_loss += total_loss.item()
        total_ot_loss += ot_loss.item()
        total_diffusion_loss += diffusion_loss.item()


    with torch.no_grad():

        if epoch % 5 == 0 or epoch == epoches - 1:
            # print(total_loss)
            # print("Epoch:" + str(epoch) + " Total Loss: " + str(total_loss.item()))
            # print(f"Epoch: {epoch} Total Loss: {total_epoch_loss:.4f} OT Loss: {total_ot_loss:.4f} Diffusion Loss: {total_diffusion_loss:.4f}")
            # logger.info("Epoch:" + str(epoch) )
            current_metric = float(logger_performance(MLA_model, spike_day_0, spike_day_k, p, q_test, test_trial_vel))
            if current_metric > best_metric:
                best_metric = current_metric
            
            # if total_loss < pre_total_loss_:
            torch.save(MLA_model.state_dict(),'../model_checkpoints/vae_model_mla.pth')
            pre_total_loss_ = total_loss 


            # Testing Phase
            _, _, _, _, test_latents, _,_, x_after_lowd = MLA_model(spike_day_0, spike_day_k,p,q_test, train_flag = False)
            test_latents = np.array(test_latents.cpu())

            vanilla_model_dict = torch.load('../model_checkpoints/vae_model_mla.pth', weights_only=True, map_location=torch.device('cpu'))

            VAE_Readout_model = VAE_Readout_Model()
            DL_dict_keys = VAE_Readout_model.state_dict().keys()
            vanilla_model_dict_keys = vanilla_model_dict.keys()

            DL_dict_new = VAE_Readout_model.state_dict().copy()

            for key in vanilla_model_dict_keys:
                DL_dict_new[key] = vanilla_model_dict[key]

            VAE_Readout_model.load_state_dict(DL_dict_new)

            print(f"Epoch of: {epoch} Perf:")
            vel_cal(test_trial_vel, VAE_Readout_model, torch.Tensor(test_latents), x_after_lowd)

            if epoch % 100 == 0 or epoch == epoches - 1:
                # best_metric
                print(f"Best_Metric at {epoch} is : {best_metric:0.4f}")
                    

