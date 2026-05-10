import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.linalg as linalg
import torch.distributions as td

import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import ot
from utils.utils_torch import compute_gaussian_kl, get_logger


logger = get_logger(__name__)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


class VAE_MLA_Model(nn.Module):
    def __init__(self, len_trial=28, num_neurons_s=187, num_neurons_t=172):
        super(VAE_MLA_Model, self).__init__()
        # Hyper-Parameters
        self.len_trial = len_trial
        self.spike_dim_s = num_neurons_s
        self.spike_dim_t = num_neurons_t
        self.low_dim = 64
        self.latent_dim = 16
        self.vel_dim = 2
        self.encoder_n_layers = 1

        # Alignment Structure
        self.low_d_readin_s = nn.Linear(self.spike_dim_s,self.low_dim, bias=False)
        self.low_d_readin_t = nn.Linear(self.spike_dim_t, self.low_dim, bias=True)

        nn.init.eye_(self.low_d_readin_t.weight)

        # Target latent affine transform (initialized as identity)
        self.latent_align_t = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.eye_(self.latent_align_t.weight)
        nn.init.zeros_(self.latent_align_t.bias)

        # Encoder Structure
        self.encoder_rnn = nn.RNN(self.low_dim, self.low_dim, self.encoder_n_layers,
         bidirectional= False, nonlinearity = 'relu', batch_first = True)
        for name, param in self.encoder_rnn.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param,0.1)
        

        self.fc_mu_1 = nn.Linear(self.low_dim, self.latent_dim)
        # self.fc_mu_2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        self.fc_log_var_1 = nn.Linear(self.low_dim, self.latent_dim)
        # self.fc_log_var_2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Spike Decoder Structure
        self.sde_fc = nn.Linear(self.latent_dim, self.spike_dim_s)

        # Velocity Decoder Structure (2-layer MLP)
        # self.vde_fc = nn.Sequential(
        #     nn.Linear(self.latent_dim, 2, bias=False),
        # )
        self.vde_fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.low_dim),
            nn.ReLU(),
            nn.Linear(self.low_dim, 2, bias=False),
        )

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


    def forward(self, x_0, x_k, p, q, train_flag):

        # Encoder
        # x_0 day_0
        x_0 = self.low_d_readin_s(x_0)
        rnn_states_x_0, _ = self.encoder_rnn(x_0)
        mu_x_0 = self.fc_mu_1(rnn_states_x_0)
        log_var_0 = self.fc_log_var_1(rnn_states_x_0)
        
        if train_flag:
            z_0 = self.reparameterize(mu_x_0, log_var_0)
        else:
            z_0 = mu_x_0

        # x_k day_k
        x_k = self.low_d_readin_t(x_k)
        rnn_states_x_k, _ = self.encoder_rnn(x_k)
        mu_x_k = self.fc_mu_1(rnn_states_x_k)
        log_var_k = self.fc_log_var_1(rnn_states_x_k)

        if train_flag:
            z_k = self.reparameterize(mu_x_k, log_var_k)
        else:
            z_k = mu_x_k

        # reshape
        d0_align = z_0.reshape(-1, self.latent_dim)
        dk_align = z_k.reshape(-1, self.latent_dim)

        # mean
        d0_mean = d0_align.mean(dim=0)
        dk_mean = dk_align.mean(dim=0)

        # covariance
        d0_cov = torch.cov(d0_align.T)
        dk_cov = torch.cov(dk_align.T)

        # create distributions
        d0_dist = td.MultivariateNormal(loc=d0_mean, covariance_matrix=d0_cov)
        dk_dist = td.MultivariateNormal(loc=dk_mean, covariance_matrix=dk_cov)

        # compute KL divergence
        kl_div = td.kl_divergence(d0_dist, dk_dist)
        
        z_0_tide = z_0.reshape((z_0.shape[0], -1))
        z_k_tide = z_k.reshape((z_k.shape[0], -1))
        
        # Calculate_OT_Dist
        M = ot.dist(z_0_tide, z_k_tide)
        M += 1e-4

        sh_div = torch.trace(M).mean()

        # Spike Decoder
        neural_hat = self.sde_fc(z_k)

        # Velocity Decoder
        vel_latent = z_k
        vel_hat = self.vde_fc(vel_latent)

        return neural_hat, vel_hat, mu_x_k, sh_div, kl_div, log_var_k