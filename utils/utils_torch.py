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
import logging



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class SpikeDataset(Dataset):
    def __init__(self, spike_day_0, spike_day_k, labels_k=None):
        self.spike_day_0 = spike_day_0
        self.spike_day_k = spike_day_k
        self.labels_k = labels_k

    def __len__(self):
        return len(self.spike_day_0)

    def __getitem__(self, idx):
        if self.labels_k is not None:
            return self.spike_day_0[idx], self.spike_day_k[idx], self.labels_k[idx]
        return self.spike_day_0[idx], self.spike_day_k[idx]


def logger_performance(model, spike_day_0, spike_day_k, p, q_test, test_trial_vel):
    re_sp_test, vel_hat_test, _, _, _, _,_,_ = model(spike_day_0, spike_day_k, p, q_test, train_flag=False)

    y_true = test_trial_vel.reshape((-1, 2))
    y_pred = vel_hat_test.cpu().detach().numpy().reshape((-1, 2))

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("NaN detected in input data.")
        y_true = np.nan_to_num(y_true)
        y_pred = np.nan_to_num(y_pred)

    key_metric = 100 * r2_score(y_true, y_pred, multioutput='uniform_average')
    return key_metric


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(name=__name__):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


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


def compute_gaussian_kl(mu0, sigma0, mu1, sigma1, eps=1e-6):
    """
    Computes the KL divergence between two multivariate Gaussians:
    N(mu0, sigma0) and N(mu1, sigma1)
    """
    D = mu0.shape[0]
    
    # Add small value to diagonal for numerical stability
    sigma0 += eps * torch.eye(D, device=sigma0.device)
    sigma1 += eps * torch.eye(D, device=sigma1.device)

    # Inverse and determinant of sigma1
    sigma1_inv = torch.linalg.inv(sigma1)
    trace_term = torch.trace(sigma1_inv @ sigma0)

    diff = mu1 - mu0
    mahalanobis = (diff.T @ sigma1_inv @ diff).squeeze()

    log_det_sigma0 = torch.logdet(sigma0)
    log_det_sigma1 = torch.logdet(sigma1)

    kl = 0.5 * (trace_term + mahalanobis - D + log_det_sigma1 - log_det_sigma0)
    return kl