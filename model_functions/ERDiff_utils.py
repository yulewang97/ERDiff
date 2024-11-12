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

def vel_cal(test_trial_vel_tide, VAE_Readout_model, test_latents, x_after_lowd):

    test_latents = Variable(torch.from_numpy(test_latents)).float()

    with torch.no_grad():
        re_sp_test, vel_hat_test = VAE_Readout_model(test_latents, train_flag=False)

        # print(f"check the weights {VAE_Readout_model.low_d_readin_t.weight}")
        # print(f"check the latents {test_latents}")
        # print(f"check the x_after_lowd {x_after_lowd}")

        y_true = test_trial_vel_tide.reshape((-1, 2))
        y_pred = vel_hat_test.cpu().detach().numpy().reshape((-1, 2))
        # print(f"check the r2 {y_pred}")

        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            print("NaN detected in input data.")
            y_true = np.nan_to_num(y_true)
            y_pred = np.nan_to_num(y_pred)

        r_squared_value = 100 * r2_score(y_true,y_pred, multioutput='uniform_average')
        rmse_value = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # 使用 format 方法
        print("-" * 20)
        print("Aligned R-Squared: {:.4f}".format(r_squared_value), end=" | ")
        print("Aligned RMSE: {:.4f}".format(rmse_value))



def skilling_divergence(z_noisy, z_0,t):
    grad = autograd.grad(outputs=z_noisy, inputs=z_0, grad_outputs=torch.ones_like(z_noisy), retain_graph=True)[0]
    divergence = torch.mean(z_noisy * grad * eps)

    return divergence