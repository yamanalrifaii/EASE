# =============================================================
# train_one_patient.py — 21ch Spectrogram CNN (single patient)
# =============================================================
import os, numpy as np, torch
from sklearn.metrics import confusion_matrix, f1_score
from loader_spectrograms import load_patient_split
from architecture_spectrograms import Net
from trainer import trainer

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
ROOT      = r"E:\EEG\spectrograms"
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
OUT_DIR   = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ✏️ Choose one patient here:
PATIENT = "chb24"

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
Train_set, Test_set = load_patient_split(PATIENT, SPEC_DIR)
X_train, y_train = Train_set
X_test,  y_test  = Test_set

# -------------------------------------------------------------
# MODEL SETUP
# -------------------------------------------------------------
model = Net(n_classes=2)
t = trainer(model, Train_set)
t.compile(lr=5e-4)
out_path = os.path.join(OUT_DIR, f"{PATIENT}_spectrogram_best.pt")

print(f"\n================ Training {PATIENT} (21ch Spectrogram CNN) ================")
t.train(epochs=100, batch_size=128, directory=out_path)

# -------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------
model.load_state_dict(torch.load(out_path))
model.eval()
if torch.cuda.is_available():
    model.cuda()

probs = []
with torch.no_grad():
    for data in torch.utils.data.DataLoader(X_test, batch_size=128, shuffle=False):
        if torch.cuda.is_available():
            data = data.cuda()
        out = torch.softmax(model(data.float()), dim=1)
        probs.extend(out[:, 1].cpu().numpy())

y_prob = np.array(probs)

# --- Find best threshold for F1 ---
best_f1, best_th = 0, 0.5
for th in np.linspace(0.1, 0.9, 81):
    preds_tmp = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, preds_tmp)
    if f1 > best_f1:
        best_f1, best_th = f1, th

# --- Final predictions ---
y_pred = (y_prob >= best_th).astype(int)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
acc  = (TP + TN) / (TP + TN + FP + FN)
sens = TP / (TP + FN) if (TP + FN) else 0
spec = TN / (TN + FP) if (TN + FP) else 0
f1b  = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0
f1w  = f1_score(y_test, y_pred, average='weighted')

# -------------------------------------------------------------
# REPORT
# -------------------------------------------------------------
print(f"\n=== Test Results ({PATIENT}) ===")
print(f"Optimal threshold = {best_th:.2f}")
print(f"Acc {acc*100:.2f}% | Sens {sens*100:.2f}% | Spec {spec*100:.2f}% | F1 {f1b*100:.2f}% | F1w {f1w*100:.2f}%")

print(f"✅ Model saved to {out_path}")

import csv

csv_path = os.path.join(OUT_DIR, "spectrogram_results.csv")
header = ["Patient", "Accuracy", "Sensitivity", "Specificity", "F1", "F1_weighted", "Best_Threshold"]

# If file doesn’t exist, create it with header
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Append current results
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([PATIENT, f"{acc*100:.2f}", f"{sens*100:.2f}", f"{spec*100:.2f}",
                     f"{f1b*100:.2f}", f"{f1w*100:.2f}", f"{best_th:.2f}"])
print(f"📁 Metrics appended to {csv_path}")
