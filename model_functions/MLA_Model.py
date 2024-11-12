import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.linalg as linalg

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm_notebook
import ot


from torch.optim import Adam

import numpy as np

len_trial, num_neurons_s, num_neurons_t = 14, 187, 172

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

class VAE_MLA_Model(nn.Module):
    def __init__(self):
        super(VAE_MLA_Model, self).__init__()
        # Hyper-Parameters
        self.spike_dim_s = num_neurons_s
        self.spike_dim_t = num_neurons_t
        self.low_dim = 64
        self.latent_dim = 8
        self.vel_dim = 2
        self.encoder_n_layers, self.decoder_n_layers = 1,1
        self.hidden_dims = [64,32]
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        # Alignment Structure
        self.low_d_readin_s = nn.Linear(self.spike_dim_s,self.low_dim)

        self.align_layer = nn.Linear(self.spike_dim_t,self.spike_dim_t, bias = False)

        self.low_d_readin_t_1 = nn.Linear(self.spike_dim_t, self.spike_dim_t)
        
        self.low_d_readin_t_2 = nn.Linear(self.spike_dim_t, self.low_dim)

        
        # self.al_fc2 = nn.Linear(self.spike_dim, self.spike_dim)

        nn.init.eye_(self.align_layer.weight)
        nn.init.eye_(self.low_d_readin_t_1.weight)
        nn.init.eye_(self.low_d_readin_t_2.weight)

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
        self.sde_fc1 = nn.Linear(self.latent_dim, self.hidden_dims[0])
        self.sde_fc2 = nn.Linear(self.hidden_dims[0], self.spike_dim_s)


        # Velocity Decoder Structure
        self.vde_rnn = nn.RNN(self.latent_dim, self.latent_dim, self.decoder_n_layers, bidirectional= False,
         nonlinearity = 'tanh', batch_first = True)
        for name, param in self.vde_rnn.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param,0.1)

        self.vde_fc_minus_0 = nn.Linear(self.latent_dim, 2, bias = False)
        self.vde_fc_minus_1 = nn.Linear(self.latent_dim, 2, bias = False)
        self.vde_fc_minus_2 = nn.Linear(self.latent_dim, 2, bias = False)

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


    def calculate_2_wasserstein_dist(self, X, Y):

        # the linear algebra ops will need some extra precision -> convert to double
        X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
        mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
        n, b = X.shape
        fact = 1.0 if b < 2 else 1.0 / (b - 1)

        # Cov. Matrix
        E_X = X - mu_X
        E_Y = Y - mu_Y
        cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
        cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

        # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
        # The eigenvalues for M are real-valued.
        C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
        C_Y = E_Y * math.sqrt(fact)
        M_l = torch.matmul(C_X.t(), C_Y)
        M_r = torch.matmul(C_Y.t(), C_X)
        M = torch.matmul(M_l, M_r)
        S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
        sq_tr_cov = S.sqrt().abs().sum()

        # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
        trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

        # |mu_X - mu_Y|^2
        diff = mu_X - mu_Y  # [n, 1]
        mean_term = torch.sum(torch.mul(diff, diff))  # scalar

        # put it together
        return (trace_term + mean_term).float()


    def forward(self, x_0, x_k, p, q, train_flag):

       # Encoder
        # x_0
        x_0 = self.low_d_readin_s(x_0)
        rnn_states_x_0, _ = self.encoder_rnn(x_0)
        mu_x_0 = self.fc_mu_1(rnn_states_x_0)
        log_var_0 = self.fc_log_var_1(rnn_states_x_0)
        
        if train_flag:
            z_0 = self.reparameterize(mu_x_0, log_var_0)
        else:
            z_0 = mu_x_0

        latent_states_x_0_tide = z_0.reshape((z_0.shape[0], -1))


        # x_k
        # x_k = self.align_layer(x_k)
        # x_after_lowd = self.elu(self.low_d_readin_t_1(x_k))
        x_after_lowd = self.low_d_readin_t_2(x_k)
        rnn_states_x_k, _ = self.encoder_rnn(x_after_lowd)
        mu_x_k = self.fc_mu_1(rnn_states_x_k)
        log_var_k = self.fc_log_var_1(rnn_states_x_k)
        # mu_x_k = self.fc_mu_2(mu_x_k)
        latent_states_x_k_tide = mu_x_k.reshape((mu_x_k.shape[0], -1))

        if train_flag:
            z_k = self.reparameterize(mu_x_k, log_var_k)
        else:
            z_k = mu_x_k
        
        latent_states_x_k_tide = z_k.reshape((z_k.shape[0], -1))

        dist_0 = torch.mean(latent_states_x_0_tide, dim = 0)
        dist_k = torch.mean(latent_states_x_k_tide, dim = 0)
    
        # kl_loss = nn.KLDivLoss()
        # output_kl_loss = kl_loss(F.log_softmax(dist_0, dim=-1), F.softmax(dist_k, dim=-1))

        X = latent_states_x_0_tide.t()
        Y = latent_states_x_k_tide.t()


        mu_0 = torch.mean(X, dim = 1, keepdim=True)
        mu_k = torch.mean(Y, dim = 1, keepdim=True)


        sigma_0 = 1 / x_0.shape[0] * torch.mm(X - mu_0, (X - mu_0).t())
        sigma_k = 1 / x_k.shape[0] * torch.mm(Y - mu_k, (Y - mu_k).t())

        
        # Calculate_OT_Dist
        x2, y2 = torch.sum(torch.pow(X, 2), dim=0), torch.sum(torch.pow(Y, 2), dim=0)
        C = (torch.tile(y2[None, :], (X.shape[1], 1)) + torch.tile(x2[:, None], (1, Y.shape[1])) - 2 * (X.T @ Y))
        M = ot.dist(X.T, Y.T)
        M += 1e-4
        # with torch.no_grad():
        # T = ot.emd(torch.squeeze(p), torch.squeeze(q), M) # exact linear program
        sh_d = ot.sinkhorn2(torch.squeeze(p), torch.squeeze(q), M/M.max(), reg=0.01) # exact linear program

        # Spike Decoder

        re_sp, _ = self.sde_rnn(z_k)
        re_sp = self.sde_fc1(re_sp)
        re_sp = self.softplus(self.sde_fc2(re_sp))
        # Velocity Decoder
        vel_latent = z_k
        vel_hat_minus_0 = self.vde_fc_minus_0(vel_latent)
        vel_hat_minus_1 = self.vde_fc_minus_1(vel_latent)
        vel_hat_minus_2 = self.vde_fc_minus_2(vel_latent)

        vel_hat = torch.zeros_like(vel_hat_minus_0)

        for i in range(len_trial):
            vel_hat[:,i,:] += vel_hat_minus_0[:,i,:]

        return re_sp, vel_hat, dist_0, dist_k, mu_x_k, sh_d,log_var_k, x_after_lowd