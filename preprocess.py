import torch
from torch.utils.data import Dataset
import numpy as np
import glob

class LightCurveDataset(Dataset):
    def __init__(self, path):
        self.files = glob.glob(f"{path}/*.npz")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        flux = data["flux"]

        flux = (flux - np.mean(flux)) / np.std(flux)

        max_len = 2000
        if len(flux) > max_len:
            flux = flux[:max_len]
        else:
            flux = np.pad(flux, (0, max_len - len(flux)))

        flux = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(data["label"], dtype=torch.long)
        return flux, label

