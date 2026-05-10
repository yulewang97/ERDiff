import scipy.io as sio
import os
from scipy.signal.windows import gaussian
import sys
import argparse
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.diffusion import Diff_STDiT, q_sample, diff_timesteps
from models.mla_model import VAE_MLA_Model
from utils.utils_torch import setup_seed, SpikeDataset, get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Set hyperparameters from command line")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--appro_alpha", type=float, default=0.0, help="Approximator alpha parameter")
    parser.add_argument("--kl_weight", type=float, default=0.0, help="Weight for KL loss")
    parser.add_argument("--ot_weight", type=float, default=0.8, help="Weight for optimal transport loss")
    parser.add_argument("--epoches", type=int, default=400, help="Alternative epoch count (possible typo in config)")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    return vars(parser.parse_args())


def alignment():

    config = parse_args()
    logger.info(f"Config Data: {config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    with open('../datasets/source_data_inte14_aligned.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('../datasets/target_data_inte14_aligned.pkl', 'rb') as f:
        test_data = pickle.load(f)

    source_trial_spikes = train_data['neural']
    target_trial_spikes, target_trial_vel = test_data['neural'], test_data['pos']
    target_labels = torch.tensor(test_data['label'], dtype=torch.long)

    num_neurons_s = source_trial_spikes.shape[-1]
    num_neurons_t = target_trial_spikes.shape[-1]

    len_trial = source_trial_spikes.shape[1]

    eps = 1e-6

    vel_mean = target_trial_vel.mean(axis=(0, 1), keepdims=True)  # shape: (1, 1, 2)
    vel_std  = target_trial_vel.std(axis=(0, 1), keepdims=True)   # shape: (1, 1, 2)
    target_trial_vel_array = (target_trial_vel - vel_mean) / (vel_std + eps)


    bin_width = float(0.02) * 1000
    kern_sd_ms = float(0.02) * 1000 * 5
    kern_sd = int(round(kern_sd_ms / bin_width))
    window = gaussian(kern_sd, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')

    source_trial_spikes_smoothed = np.apply_along_axis(filt, 1, source_trial_spikes)
    target_trial_spikes_smoothed = np.apply_along_axis(filt, 1, target_trial_spikes)

    # Use unconditional diffusion model for guidance in MLA alignment
    diff_model = Diff_STDiT(input_dim=1).to(device)

    diff_model_dict = torch.load(
        '../model_checkpoints/latent_diffusion_t300.pth',
        map_location=torch.device('cpu'),
        weights_only=True,
    )
    diff_model.load_state_dict(diff_model_dict)

    for k,v in diff_model.named_parameters():
        v.requires_grad=False

    setup_seed(config["seed"])
    MLA_model = VAE_MLA_Model(len_trial=len_trial, num_neurons_s=num_neurons_s, num_neurons_t=num_neurons_t).to(device)
    vanilla_model_dict = torch.load('../model_checkpoints/source_vae_inte14_v6_d16.pt', weights_only=True, map_location=torch.device('cpu'))

    MLA_dict_new = MLA_model.state_dict().copy()

    for key, value in vanilla_model_dict.items():
        if key in MLA_dict_new:
            MLA_dict_new[key] = value

    MLA_model.load_state_dict(MLA_dict_new)

    # Then manually copy a subset of parameters
    with torch.no_grad():
        s_weight = MLA_model.low_d_readin_s.weight  # Shape: [low_dim, spike_dim_s]

        # spike_dim_t equals the in_features of low_d_readin_t
        target_input_dim = MLA_model.low_d_readin_t.weight.size(1)

        # Slice safely to match target input dim
        sliced_weight = s_weight[:, :target_input_dim]

        # Copy the sliced weights
        MLA_model.low_d_readin_t.weight.copy_(sliced_weight)

    best_r_squared = -float('inf')
    l_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    ot_weight = config["ot_weight"]
    appro_alpha = config["appro_alpha"]
    kl_weight = config["kl_weight"]
    epoches = config["epoches"]
    logger.info(f"Total epochs: {epoches}")

    optimizer = torch.optim.Adam(MLA_model.parameters(), lr=l_rate)

    # Freeze the other parameters
    for param in MLA_model.parameters():
        param.requires_grad = False

    # Only unfreeze the target session Input Read-in layer and latent affine transform
    for param in MLA_model.low_d_readin_t.parameters():
        param.requires_grad = True

    spike_day_0 = torch.from_numpy(source_trial_spikes_smoothed).float().to(device)
    spike_day_k = torch.from_numpy(target_trial_spikes_smoothed).float().to(device)
    spike_dataset = SpikeDataset(spike_day_0, spike_day_k, labels_k=target_labels.to(device))

    logger.info(f'spike_day_0 shape: {spike_day_0.shape}, spike_day_k shape: {spike_day_k.shape}')

    dataloader = DataLoader(spike_dataset, batch_size=batch_size, shuffle=True)

    num_day_0, num_day_k = spike_day_0.shape[0], spike_day_k.shape[0]

    logger.info(f"num_day_0: {num_day_0}, num_day_k: {num_day_k}")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_name = f'ERDiff_MLA_{timestamp}'

    # Maximum Likelihood Alignment
    for epoch in range(epoches):
        current_epoch = epoch + 1
        total_epoch_loss = 0
        total_ot_loss = 0
        total_kl_loss = 0
        total_diffusion_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()
            spike_day_0_batch, spike_day_k_batch, labels_k_batch = batch
            cur_batch_size = spike_day_0_batch.shape[0]

            p = torch.full((cur_batch_size, 1), 1 / cur_batch_size, device=device, dtype=torch.float32)
            q = torch.full((cur_batch_size, 1), 1 / cur_batch_size, device=device, dtype=torch.float32)

            _, _, latents_k, output_sh_loss, output_kl_loss, _ = MLA_model(spike_day_0_batch, spike_day_k_batch, p, q, train_flag=False)

            ot_loss = ot_weight * output_sh_loss
            kl_loss = kl_weight * output_kl_loss

            latents_k = latents_k[:, None, :, :]

            batch_size = latents_k.shape[0]
            t = torch.randint(0, diff_timesteps // 6, (batch_size,), device=device).long()
            noise = torch.randn_like(latents_k, device=device)

            z_noisy = q_sample(x_start=latents_k, t=t, noise=noise)
            # unconditional guidance: does not pass labels to the diffusion model here
            predicted_noise = diff_model(z_noisy, t)
            diffusion_loss = appro_alpha * (0.5 * F.l1_loss(predicted_noise, noise) + 0.5 * F.mse_loss(predicted_noise, noise))
            
            loss = kl_loss + diffusion_loss

            loss.backward(retain_graph=True)
            
            optimizer.step()

            total_epoch_loss += loss.item()
            total_ot_loss += ot_loss.item()
            total_kl_loss += kl_loss.item()
            total_diffusion_loss += diffusion_loss.item()
        with torch.no_grad():
            if current_epoch % 25 == 0 or current_epoch == 1:
                p = torch.full((num_day_0, 1), 1 / num_day_0, device=device, dtype=torch.float32)
                q = torch.full((num_day_k, 1), 1 / num_day_k, device=device, dtype=torch.float32)

                # Testing Phase
                _, test_vel_hat, _, _, _, _ = MLA_model(spike_day_0, spike_day_k, p, q, train_flag = False)

                logger.info(f"Epoch of: {current_epoch} Performance:")
                y_true = torch.as_tensor(target_trial_vel_array, dtype=torch.float32, device=device).reshape((-1, 2))
                y_pred = test_vel_hat.detach().reshape((-1, 2))

                ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
                ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0, keepdim=True)) ** 2, dim=0).clamp_min(1e-12)
                r_squared_value = 100 * torch.mean(1 - ss_res / ss_tot).item()
                rmse_value = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

                logger.info("Aligned R-Squared: {:.3f} | Aligned RMSE: {:.3f}".format(r_squared_value, rmse_value))
                logger.info("-" * 20)

                if r_squared_value > best_r_squared:
                    best_r_squared = r_squared_value


    logger.info(f"Best_r_squared is : {best_r_squared:.3f}")

if __name__ == "__main__":
    alignment()