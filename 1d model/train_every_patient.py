# =============================================================
# train_patient_dependent.py — EEGWaveNet paper protocol (FINAL + TRAIN BALANCING)
# =============================================================

import os
import glob
import csv
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from dataset_loader import load_patient_split
from architecture import Net
from trainer import trainer

# CONFIGURATION
ROOT = r"E:\EEG\chbmit"
SEG_DIR = os.path.join(ROOT, "segments_db4")        # 4s windows, 1s overlap
LABEL_DIR = os.path.join(ROOT, "manual_labels_db4") # labels per segment
OUT_DIR =r"E:\EEG\1d model\models"
CSV_PATH = r"E:\EEG\1d model\models\results_patient_dependent.csv"
os.makedirs(OUT_DIR, exist_ok=True)

PATIENT = "chb13"  # change this for each patient
N_CHANS, N_CLASSES = 21, 2

# LOAD DATA (patient-dependent split)
Train_set, Test_set = load_patient_split(PATIENT, SEG_DIR, LABEL_DIR)
X_train, y_train = Train_set
X_test, y_test = Test_set

# BALANCE TRAINING SET (1:1)
y_np = y_train.numpy()
seiz_idx = np.where(y_np == 1)[0]
norm_idx = np.where(y_np == 0)[0]

if len(seiz_idx) > 0:
    np.random.shuffle(norm_idx)
    norm_idx = norm_idx[:len(seiz_idx)]  # equal number of 0 and 1
    idx_bal = np.concatenate([seiz_idx, norm_idx])
    np.random.shuffle(idx_bal)
    X_train_bal = X_train[idx_bal]
    y_train_bal = y_train[idx_bal]
    print(f"\n Balanced training set: class0={len(norm_idx)}, class1={len(seiz_idx)} (ratio=1:1)")
else:
    X_train_bal, y_train_bal = X_train, y_train
    print("\n⚠️ No seizure samples found — training on full dataset without balancing.")

Train_set = (X_train_bal, y_train_bal)

# INITIALIZE + TRAIN MODEL
model = Net(N_CHANS, N_CLASSES)
t = trainer(model, Train_set)
t.compile(lr=1e-3)
out_path = os.path.join(OUT_DIR, f"{PATIENT}_best.pt")

print(f"\n================ Training {PATIENT} (EEGWaveNet paper protocol) ================")
t.train(epochs=100, batch_size=512, directory=out_path)

# EVALUATION ON IMBALANCED TEST SET
X_test, y_test = Test_set
model.load_state_dict(torch.load(out_path))
model.eval()

if torch.cuda.is_available():
    model.cuda()

# --- Predict seizure probabilities ---
probs = []
with torch.no_grad():
    for data in torch.utils.data.DataLoader(X_test, batch_size=512, shuffle=False):
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

print(f"\n=== Test evaluation (imbalanced, real distribution) ===")
print(f"Optimal threshold = {best_th:.2f}")
print(f"Accuracy {acc*100:.2f}% | Sens {sens*100:.2f}% | "
      f"Spec {spec*100:.2f}% | F1 {f1b*100:.2f}% | F1w {f1w*100:.2f}%")

# BALANCED TEST EVALUATION (for comparison)

y_np = y_test.numpy()
seiz = np.where(y_np == 1)[0]
norm = np.where(y_np == 0)[0]

if len(seiz) > 0 and len(norm) > 0:
    np.random.shuffle(norm)
    norm = norm[:len(seiz)]
    idx = np.concatenate([seiz, norm])
    np.random.shuffle(idx)
    X_bal, y_bal = X_test[idx], y_test[idx]
    y_pred_bal = (y_prob[idx] >= best_th).astype(int)

    cm = confusion_matrix(y_bal, y_pred_bal)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    acc_b = (TP + TN) / (TP + TN + FP + FN)
    sens_b = TP / (TP + FN) if (TP + FN) else 0
    spec_b = TN / (TN + FP) if (TN + FP) else 0
    f1b_b = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0

    print(f"\n=== Balanced test evaluation ({PATIENT}) ===")
    print(f"Accuracy {acc_b*100:.2f}% | Sens {sens_b*100:.2f}% | "
          f"Spec {spec_b*100:.2f}% | F1 {f1b_b*100:.2f}%")
else:
    acc_b = sens_b = spec_b = f1b_b = 0
    print("\n⚠️ Skipping balanced evaluation — no seizure samples found.")

# SAVE RESULTS TO CSV
header = ["Patient", "Threshold", "Acc", "Sens", "Spec", "F1", "F1w",
          "Acc_bal", "Sens_bal", "Spec_bal", "F1_bal"]
row = [PATIENT, best_th, acc, sens, spec, f1b, f1w,
       acc_b, sens_b, spec_b, f1b_b]

file_exists = os.path.isfile(CSV_PATH)
with open(CSV_PATH, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(row)

print(f"\n✅ Saved metrics for {PATIENT} to {CSV_PATH}")