import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import random
from sklearn.metrics import r2_score
import sys
from torch.utils.data import Dataset


timesteps = 50
eps = 1 / timesteps


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class SpikeDataset(Dataset):
    def __init__(self, spike_day_0, spike_day_k):
        self.spike_day_0 = spike_day_0
        self.spike_day_k = spike_day_k

    def __len__(self):
        return len(self.spike_day_0)

    def __getitem__(self, idx):
        return self.spike_day_0[idx], self.spike_day_k[idx]


def logger_performance(model, spike_day_0, spike_day_k, p, q_test, test_trial_vel):
    re_sp_test, vel_hat_test, _, _, _, _,_,_ = model(spike_day_0, spike_day_k, p, q_test, train_flag=False)

    y_true = test_trial_vel.reshape((-1, 2))
    y_pred = vel_hat_test.cpu().detach().numpy().reshape((-1, 2))

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("NaN detected in input data.")
        y_true = np.nan_to_num(y_true)
        y_pred = np.nan_to_num(y_pred)

    key_metric = 100 * r2_score(y_true,y_pred, multioutput='uniform_average')
    return  key_metric


def vel_cal(test_trial_vel_tide, VAE_Readout_model, test_latents, x_after_lowd):
    with torch.no_grad():
        re_sp_test, vel_hat_test = VAE_Readout_model(test_latents, train_flag=False)


        y_true = test_trial_vel_tide.reshape((-1, 2))
        y_pred = vel_hat_test.cpu().detach().numpy().reshape((-1, 2))

        assert not np.isnan(y_pred).any(), "Error: y_pred contains NaN values!"

        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            print("NaN detected in input data.")
            y_true = np.nan_to_num(y_true)
            y_pred = np.nan_to_num(y_pred)

        r_squared_value = 100 * r2_score(y_true,y_pred, multioutput='uniform_average')
        rmse_value = np.sqrt(np.mean((y_true - y_pred) ** 2))


        print("Aligned R-Squared: {:.4f}".format(r_squared_value), end=" | ")
        print("Aligned RMSE: {:.4f}".format(rmse_value))
        print("-" * 20)

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

def skilling_divergence(z_noisy, z_0,t):
    grad = autograd.grad(outputs=z_noisy, inputs=z_0, grad_outputs=torch.ones_like(z_noisy), retain_graph=True)[0]
    divergence = torch.mean(z_noisy * grad * eps)

    return divergence