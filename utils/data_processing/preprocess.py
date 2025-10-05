import torch
from torch.utils.data import Dataset
import numpy as np
import glob

class LightCurveDataset(Dataset):
    def __init__(self, path, augment=False):
        """
        path: carpeta con los .npz
        augment: True activa data augmentation on-the-fly
        """
        self.files = glob.glob(f"{path}/*.npz")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        flux = data["flux"].copy()
        label = int(data["label"])

        if self.augment:
            flux = self.augment_flux(flux, label)

        flux = (flux - np.mean(flux)) / np.std(flux)
        max_len = 2000
        if len(flux) > max_len:
            flux = flux[:max_len]
        else:
            flux = np.pad(flux, (0, max_len - len(flux)))

        flux = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return flux, label

    def augment_flux(self, flux, label):
        shift = np.random.randint(-5, 5)
        flux = np.roll(flux, shift)

        flux += np.random.normal(0, 0.0005, len(flux))

        if label in [1, 2]:
            num_transits = np.random.randint(1, 3)
            for _ in range(num_transits):
                pos = np.random.randint(300, 1700)
                depth = np.random.uniform(0.001, 0.01) if label == 1 else np.random.uniform(0.0005, 0.005)
                width = np.random.randint(5, 50)
                flux[pos:pos+width] -= depth

        return flux

