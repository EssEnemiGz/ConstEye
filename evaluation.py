import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve, auc)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader, random_split
from preprocess import LightCurveDataset
from model import ExoCNN
import torch.nn.functional as F

DATA_PATH = "data/curves"
MODEL_PATH = "models/exo_cnn.pth"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "evaluation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

dataset = LightCurveDataset(DATA_PATH)
n = len(dataset)
test_size = max(1, int(0.2 * n))
train_size = n - test_size
_, test_ds = random_split(dataset, [train_size, test_size])

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = ExoCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true = []
y_pred = []
y_probs = []
filenames = []

with torch.no_grad():
    for i, (flux, label) in enumerate(test_dl):
        flux = flux.to(DEVICE)
        logits = model(flux)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true.extend(label.numpy())
        y_pred.extend(preds.tolist())
        y_probs.extend(probs.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

print("\nClassification report (test):\n")
print(classification_report(y_true, y_pred, target_names=["No Exo", "Exo", "Candidato"], zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.xticks([0,1,2], ["No Exo","Exo","Candidato"])
plt.yticks([0,1,2], ["No Exo","Exo","Candidato"])
plt.title("Confusion matrix")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

n_classes = y_probs.shape[1]
plt.figure(figsize=(8,6))
for c in range(n_classes):
    precision, recall, _ = precision_recall_curve((y_true==c).astype(int), y_probs[:,c])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"class {c} (AUC={pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall curves")
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curves.png"))
plt.close()

plt.figure(figsize=(8,6))
for c in range(n_classes):
    fpr, tpr, _ = roc_curve((y_true==c).astype(int), y_probs[:,c])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"class {c} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],"k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC curves")
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
plt.close()

plt.figure(figsize=(8,6))
for c in range(n_classes):
    prob_true, prob_pred = calibration_curve((y_true==c).astype(int), y_probs[:,c], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f"class {c}")
plt.plot([0,1],[0,1],"k--", alpha=0.3)
plt.xlabel("Predicted probability")
plt.ylabel("Empirical probability")
plt.legend()
plt.title("Calibration curves")
plt.savefig(os.path.join(OUTPUT_DIR, "calibration.png"))
plt.close()

mis_dir = os.path.join(OUTPUT_DIR, "misclassified")
os.makedirs(mis_dir, exist_ok=True)

test_list = [test_ds[i] for i in range(len(test_ds))]

for idx, (flux_tensor, label) in enumerate(test_list):
    pred = y_pred[idx]
    true = int(y_true[idx])
    if pred != true:
        flux = flux_tensor.squeeze().numpy()
        fname = f"idx_{idx}_true{true}_pred{pred}.png"
        plt.figure(figsize=(8,2))
        plt.plot(flux)
        plt.title(f"true={true} pred={pred}")
        plt.tight_layout()
        plt.savefig(os.path.join(mis_dir, fname))
        plt.close()

print(f"\nSaved evaluation artifacts to {OUTPUT_DIR}/ (confusion, pr, roc, calibration, misclassified/)\n")

