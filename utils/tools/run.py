import torch
from model import ExoCNN
import numpy as np

data = np.load("data/curves/Kepler-9.npz")
flux = (data["flux"] - np.mean(data["flux"])) / np.std(data["flux"])
target_len = 2000
if len(flux) < target_len:
    flux = np.pad(flux, (0, target_len - len(flux)))
else:
    flux = flux[:target_len]
flux = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

model = ExoCNN()
model.load_state_dict(torch.load("models/exo_cnn.pth", map_location="cpu"))
model.eval()

pred = torch.argmax(model(flux))
if pred.item() == 1:
    print("Predicción:", "Exoplaneta")
elif pred.item() == 2:
    print("Predicción:", "Candidato")
else:
    print("Predicción:", "No exoplaneta")
