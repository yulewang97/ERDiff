import math
from inspect import isfunction
from tqdm.auto import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F
from tqdm import tqdm_notebook


# timesteps
diff_embedding_dim = 32
diff_beta_start = 0.0001
diff_beta_end = 0.5
diff_timesteps = 100

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0) 
        table = steps * frequencies  
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  
        return table


class STBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # self.cond_projection = Conv1d_with_init(diff_cond_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.temporal_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.spatio_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_temporal(self, y, base_shape):
        B, channel, K, L = base_shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.temporal_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_spatio(self, y, base_shape):
        B, channel, K, L = base_shape
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.spatio_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  
        y = x + diffusion_emb

        y = self.forward_temporal(y, base_shape)
        y = self.forward_spatio(y, base_shape) 
        y = self.mid_projection(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class Diff_STDiT(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.channels = 32
        self.diff_layers = 3
        self.diff_nheads = 4

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=diff_timesteps,
            embedding_dim=diff_embedding_dim,
        )

        self.input_projection = Conv1d_with_init(input_dim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.STBlock_layers = nn.ModuleList(
            [
                STBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=diff_embedding_dim,
                    nheads=self.diff_nheads,
                )
                for _ in range(self.diff_layers)
            ]
        )

    def forward(self, x, diffusion_step):
        B, input_dim, K, L = x.shape

        x = x.reshape(B, input_dim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.STBlock_layers:
            x, skip_connection = layer(x, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.STBlock_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, 1, K, L)
        return x


def quadratic_beta_schedule(diff_timesteps):
    beta_start = 0.0001
    beta_end = 0.5
    return torch.linspace(beta_start**0.5, beta_end**0.5, diff_timesteps) ** 2


def cosine_beta_schedule(diff_timesteps, s=0.008):
    t = torch.linspace(0, diff_timesteps, steps=diff_timesteps + 1)
    alphas_cumprod = torch.cos(((t / diff_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

betas = cosine_beta_schedule(diff_timesteps=diff_timesteps)


# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    loss = F.mse_loss(noise, predicted_noise)
    # loss = 0.7 * F.l1_loss(predicted_noise, noise) + 0.3 * F.mse_loss(predicted_noise, noise)

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]

    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, diff_timesteps)), desc='sampling loop time step', total=diff_timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    # imgs.append(img.cpu().numpy())
    return img.cpu().numpy()


class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        
        device = next(model.parameters()).device  

        self.shadow = {k: v.clone().detach().to(device) for k, v in model.named_parameters()}
    
    def update(self, model):
        with torch.no_grad():
            for k, v in model.named_parameters():
                self.shadow[k] = self.shadow[k].to(v.device)
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.clone().detach()

    def apply(self, model):
        model.load_state_dict(self.shadow, strict=False)
