from torch.utils.data import Dataset, DataLoader
import torch

class AddGaussianNoise:
    def __init__(self, factor=0.08):
        self.factor = factor

    def __call__(self, x):
        std = x.std()
        noise = torch.randn_like(x) * self.factor * std
        return x + noise


class LatentDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

        if self.labels is not None and len(self.data) != len(self.labels):
            raise ValueError(
                f"data size ({len(self.data)}) and labels size ({len(self.labels)}) must match"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        if self.labels is None:
            return sample

        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return sample, label
