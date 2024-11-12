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
import sys
from tqdm import tqdm_notebook


from torch.optim import Adam

import numpy as np

len_trial,num_neurons = 14, 187

start_pos = 0
end_pos = 1

class VAE_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Hyper-Parameters
        self.spike_dim = num_neurons
        self.low_dim = 64
        self.latent_dim = 8
        self.vel_dim = 2
        self.encoder_n_layers, self.decoder_n_layers = 1,1
        self.hidden_dims = [64,32]

        # Low-D Readin
        self.low_d_readin_s = nn.Linear(self.spike_dim,self.low_dim, bias = False)

        # Encoder Structure
        self.encoder_rnn = nn.RNN(self.low_dim, self.hidden_dims[0], self.encoder_n_layers,
         bidirectional= False, nonlinearity = 'tanh', batch_first = True)
        for name, param in self.encoder_rnn.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param,0.1)
        
        # self.encoder_rnn_bn = nn.BatchNorm1d(len_trial)

        self.fc_mu_1 = nn.Linear(self.hidden_dims[0], self.latent_dim)
        # self.fc_mu_2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        self.fc_log_var_1 = nn.Linear(self.hidden_dims[0], self.latent_dim)
        # self.fc_log_var_2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)


        # Spike Decoder Structure
        self.sde_rnn = nn.RNN(self.latent_dim, self.latent_dim, self.decoder_n_layers, bidirectional= False,
         nonlinearity = 'tanh', batch_first = True)

        # self.sde_rnn_bn = nn.BatchNorm1d(len_trial)

        self.sde_fc1 = nn.Linear(self.latent_dim, self.hidden_dims[0])
        self.sde_fc2 = nn.Linear(self.hidden_dims[0], self.spike_dim)

        # Velocity Decoder Structure
        self.vde_rnn = nn.RNN(self.latent_dim, self.latent_dim, self.decoder_n_layers, bidirectional= False,
         nonlinearity = 'tanh', batch_first = True)
        for name, param in self.vde_rnn.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param,0.1)

        self.vde_fc_minus_0 = nn.Linear(self.latent_dim, 2, bias = False)
        self.vde_fc_minus_1 = nn.Linear(self.latent_dim, 2, bias = False)
        self.vde_fc_minus_2 = nn.Linear(self.latent_dim, 2, bias = False)

        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, train_flag):
        # Encoder 

        x = self.low_d_readin_s(x)

        rnn_states, _ = self.encoder_rnn(x)

        mu = self.fc_mu_1(rnn_states)
        # mu = self.fc_mu_2(mu)

        log_var = self.fc_log_var_1(rnn_states)
        # log_var = self.fc_log_var_2(log_var)

        if train_flag:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        # Spike Decoder
        re_sp, _ = self.sde_rnn(z)
        re_sp = self.sde_fc1(re_sp)
        re_sp = (self.sde_fc2(re_sp))
        # re_sp = self.softplus(self.sde_fc2(re_sp))

        # Velocity Decoder
        vel_latent = z
        vel_hat_minus_0 = self.vde_fc_minus_0(vel_latent)
        vel_hat_minus_1 = self.vde_fc_minus_1(vel_latent)
        vel_hat_minus_2 = self.vde_fc_minus_2(vel_latent)

        vel_hat = torch.zeros_like(vel_hat_minus_0)
        # print("len_trial:", len_trial)
        # print("start_pos:", start_pos)
        for i in range(len_trial - start_pos):
            vel_hat[:,i,:] += vel_hat_minus_0[:,i,:]
            # if i > 0:
            #     vel_hat[:,i,:] += vel_hat_minus_1[:,i-1,:]
            # if i > 1:
            #     vel_hat[:,i,:] += vel_hat_minus_2[:,i-2,:]

        return re_sp, vel_hat, mu, log_var