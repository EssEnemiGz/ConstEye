import torch
from torch.utils.data import DataLoader, random_split
from preprocess import LightCurveDataset
from model import ExoCNN
from collections import Counter
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np

dataset = LightCurveDataset("data/curves", augment=True)

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"

counts = Counter([int(label.item()) if torch.is_tensor(label) else int(label) for _, label in dataset])
total = sum(counts.values())
num_classes = 3

weights = torch.tensor(
    [total / (counts[i] if counts[i] > 0 else 1) for i in range(num_classes)],
    dtype=torch.float32
)
weights = weights / weights.sum()
weights = weights.to(device)

print("Class counts:", counts)
print("Loss weights:", weights)

criterion = nn.CrossEntropyLoss(weight=weights)

model = ExoCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Entrenamiento
best_loss = float("inf")
patience = 5
trigger_times = 0

for epoch in range(50):
    model.train()
    total_loss = 0
    for flux, label in train_dl:
        flux, label = flux.to(device), label.to(device)
        optimizer.zero_grad()
        preds = model(flux)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dl)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
        torch.save(model.state_dict(), "models/exo_cnn.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break
# Evaluación
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for flux, label in test_dl:
        flux = flux.to(device)
        out = model(flux)
        preds = torch.argmax(out, dim=1)
        y_true.extend(label.numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=["No Exo", "Exo", "Candidato"]))
torch.save(model.state_dict(), "models/exo_cnn.pth")

