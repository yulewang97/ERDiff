import numpy as np 
import scipy.ndimage
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score
from torch import autograd

timesteps = 50
eps = 1 / timesteps

def get_batches(x,batch_size):
    n_batches = len(x)//(batch_size)
    x = x[:n_batches*batch_size:]
    for n in range(0, x.shape[0],batch_size):
        x_batch = x[n:n+(batch_size)]
        yield x_batch

def vel_cal(test_trial_vel_tide, VAE_Readout_model, test_latents):

    test_latents = Variable(torch.from_numpy(test_latents)).float()

    with torch.no_grad():
        re_sp_test, vel_hat_test = VAE_Readout_model( test_latents, train_flag=False)

        print("Aligned R**2:" + str(100 * r2_score(test_trial_vel_tide.reshape((-1,2)),vel_hat_test.reshape((-1,2)), multioutput='uniform_average')))
        print("Aligned RMSE:" + str(np.sqrt(mean_squared_error(test_trial_vel_tide.reshape((-1,2)),vel_hat_test.reshape((-1,2))))))


def skilling_divergence(z_noisy, z_0,t):
    grad = autograd.grad(outputs=z_noisy, inputs=z_0, grad_outputs=torch.ones_like(z_noisy), retain_graph=True) [0]
    divergence = torch.mean(z_noisy * grad * eps)

    return divergence