"""
ERDiff Utilities Module

This module provides utility constants and helper functions for the ERDiff model training.
It contains hyperparameters and configuration values used across different training scripts.
"""

import torch
import torch.nn.functional as F
import math

# Training hyperparameters
n_epochs = 500
timesteps = 100

# Diffusion model hyperparameters
diff_layers = 4
diff_channels = 32
diff_nheads = 4
diff_embedding_dim = 32
diff_beta_start = 0.0001
diff_beta_end = 0.5
diff_num_steps = 200
diff_cond_dim = 144


def quadratic_beta_schedule(timesteps):
    """
    Create a quadratic beta schedule for diffusion.
    
    Args:
        timesteps: Number of diffusion timesteps
        
    Returns:
        Beta values as a torch tensor
    """
    beta_start = 0.0001
    beta_end = 0.5
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Create a cosine beta schedule for diffusion.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small near t=0
        
    Returns:
        Beta values as a torch tensor
    """
    t = torch.linspace(0, timesteps, steps=timesteps + 1)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """
    Extract values from tensor a at indices t and reshape to match x_shape.
    
    Args:
        a: Source tensor
        t: Indices tensor
        x_shape: Target shape
        
    Returns:
        Extracted and reshaped tensor
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, noise=None, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None):
    """
    Forward diffusion process - add noise to data.
    
    Args:
        x_start: Original data
        t: Timestep
        noise: Optional noise tensor (generated if not provided)
        sqrt_alphas_cumprod: Precomputed cumulative product of alphas
        sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1 - cumulative product of alphas)
        
    Returns:
        Noised data at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    if sqrt_alphas_cumprod is None or sqrt_one_minus_alphas_cumprod is None:
        # Use default schedules if not provided
        betas = cosine_beta_schedule(timesteps=diff_num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None):
    """
    Calculate the loss for the denoising model.
    
    Args:
        denoise_model: The denoising neural network
        x_start: Original data
        t: Timestep
        noise: Optional noise tensor
        
    Returns:
        Loss value
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    
    loss = F.mse_loss(noise, predicted_noise)
    return loss
