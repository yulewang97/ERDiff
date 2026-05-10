import torch
import torch.nn.functional as F

import numpy as np
import os
import sys
import argparse
import pickle
from datetime import datetime
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.diffusion import *
from models.dataloader import LatentDataset
from utils.utils_torch import get_logger, setup_seed
from torchvision.transforms import ToTensor


logger = get_logger(__name__)


def train(batch_size=16, n_epochs=800):
    setup_seed(2026)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"Diffusion_Train_{timestamp}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loading
    dataset_path = "../datasets/source_latents_aug_dataset.pkl"
    with open(dataset_path, "rb") as f:
        source_dataset = pickle.load(f)

    source_latents = np.asarray(source_dataset["latents"], dtype=np.float32)
    source_labels = np.asarray(source_dataset["labels"], dtype=np.int64)

    # Unconditional diffusion: do not use labels for conditioning.
    dm_model = Diff_STDiT(input_dim=1, num_classes=None)
    dm_model.to(device)

    ema = EMA(dm_model, decay=0.98)

    dm_optimizer = AdamW(dm_model.parameters(), lr=3e-4, weight_decay=1e-4)

    logger.info(f"source_latents shape: {source_latents.shape}")
    logger.info(f"source_labels shape: {source_labels.shape}")
    logger.info(f"Min: {np.min(source_latents):.6f}, Max: {np.max(source_latents):.6f}")
    logger.info(f"Mean: {np.mean(source_latents):.6f}, Std: {np.std(source_latents):.6f}")

    transform = transforms.Compose([
        ToTensor(),
    ])

    dataset = LatentDataset(source_latents, labels=source_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ckpt_name = (
        f"latent_diffusion_t{diff_timesteps}.pth"
    )
    ckpt_path = os.path.join("../model_checkpoints", ckpt_name)

    for epoch in range(n_epochs):
        total_loss = 0
        total_steps = 0
        for step, batch_data in enumerate(dataloader):
            dm_optimizer.zero_grad()

            batch, labels = batch_data
            curr_batch_size = batch.shape[0]
            batch = batch.to(device)
            labels = labels.to(device)

            t = torch.randint(0, diff_timesteps, (curr_batch_size,), device=device).long()

            loss = p_losses(dm_model, batch, t)
            total_loss += loss.item()
            total_steps += 1

            loss.backward()
            dm_optimizer.step()

            ema.update(dm_model)  # update EMA parameters

        avg_epoch_loss = total_loss / max(total_steps, 1)
        
        if (epoch + 1) % 25 == 0:
            logger.info(f"Epoch {epoch + 1}: avg_epoch_loss={avg_epoch_loss:.4f}")
            logger.info(f"Epoch {epoch + 1}: total_epoch_loss={total_loss:.4f}")

            raw_state = {k: v.detach().cpu().clone() for k, v in dm_model.state_dict().items()}

            ema.apply(dm_model)  # apply EMA parameters to the model
            torch.save(dm_model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            dm_model.load_state_dict(raw_state)
            dm_model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=800)
    args = parser.parse_args()
    train(batch_size=args.batch_size, n_epochs=args.n_epochs)