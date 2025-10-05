import torch
from torch.utils.data import DataLoader, random_split
from preprocess import LightCurveDataset
from model import ExoCNN
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

dataset = LightCurveDataset("data/curves")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=8)
device = "cuda" if torch.cuda.is_available() else "cpu"

counts = Counter([label for _, label in dataset])
total = sum(counts.values())

num_classes = 3
weights = []

for i in range(num_classes):
    if counts[i] > 0:
        weights.append(total / counts[i])
    else:
        weights.append(0.0)

weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum()
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

criterion = nn.CrossEntropyLoss(weight=weights)

model = ExoCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
for epoch in range(100):
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
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f}")

# Evaluación
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for flux, label in test_dl:
        flux = flux.to(device)
        out = model(flux)
        preds = torch.argmax(F.softmax(out, dim=1), dim=1)
        y_true.extend(label.numpy())
        y_pred.extend(preds.cpu().numpy())

print("Class counts:", counts)
print("Loss weights:", weights)
print(classification_report(y_true, y_pred, target_names=["No Exo", "Exo", "Candidato"]))
torch.save(model.state_dict(), "models/exo_cnn.pth")

