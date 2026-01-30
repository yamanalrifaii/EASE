# =============================================================
# train_mob_lstm.py — MobileNetV2 + LSTM Training Script
# =============================================================
import os, csv, numpy as np, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score

from loader_spectrograms import load_patient_split
from architecture_mob import Net

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
ROOT      = r"E:\EEG\TRANSFER"
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
OUT_DIR   = os.path.join(ROOT, "models"); os.makedirs(OUT_DIR, exist_ok=True)

PATIENT    = "chb24"   # <<< change per patient
SEQ_LEN    = 3
TEST_SIZE  = 0.3
EPOCHS     = 150
BATCH_SIZE = 32
LR         = 3e-5
OVERSAMPLE = 6

device = "cuda" if torch.cuda.is_available() else "cpu"

# Weighted CE (favor seizures)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 3.0], device=device)
)

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def oversample(X, y, factor=OVERSAMPLE):
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    pos = np.where(y_np == 1)[0]; neg = np.where(y_np == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    pos_rep = np.tile(pos, factor)
    idx = np.concatenate([neg, pos_rep])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc  = (TP+TN)/(TP+TN+FP+FN+1e-8)
    sens = TP/(TP+FN+1e-8)
    spec = TN/(TN+FP+1e-8)
    f1   = 2*TP/(2*TP+FP+FN+1e-8)
    f1w  = f1_score(y_true, y_pred, average='weighted') if cm.size==4 else 0
    bal  = (sens+spec)/2
    return acc,sens,spec,f1,f1w,bal

def predict_proba(model, X, batch=256):
    model.eval(); probs=[]
    loader = DataLoader(X, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.float())
            p = torch.softmax(logits, dim=1)[:,1]
            probs.extend(p.cpu().numpy())
    return np.array(probs)

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # === Load data with sequences ===
    (X_train, y_train), (X_test, y_test) = load_patient_split(
        PATIENT, SPEC_DIR, test_size=TEST_SIZE, augment=True, seq_len=SEQ_LEN
    )

    # Oversample seizures
    X_train, y_train = oversample(X_train, y_train, OVERSAMPLE)

    # Datasets & loaders
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Model
    model = Net(n_classes=2, seq_len=SEQ_LEN, hidden_size=256, fine_tune=True).to(device)

    # Optimizer + scheduler
    opt   = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)

    # ---------------- Stage: Train -----------------
    for e in range(EPOCHS):
        model.train(); losses=[]
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = model(Xb.float())
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
        sched.step()
        print(f"Epoch {e:03d} | Train {torch.mean(torch.stack(losses)):.5f}")

    out_path = os.path.join(OUT_DIR, f"{PATIENT}_MobileNetV2LSTM.pt")
    torch.save(model.state_dict(), out_path)
    print(f"✅ Training complete. Model saved to {out_path}")

    # ---------------- Evaluation -------------------
    y_prob = predict_proba(model, X_test)
    y_true = y_test.numpy()

    best_j, best_th = -1, 0.3
    for th in np.linspace(0.05, 0.6, 112):
        acc, sens, spec, _, _, _ = compute_metrics(y_true, y_prob, th)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_th = j, th

    acc, sens, spec, f1, f1w, bal = compute_metrics(y_true, y_prob, best_th)
    print(f"\n=== Final Test Results ({PATIENT}, MobileNetV2-LSTM) ===")
    print(f"Threshold = {best_th:.2f}")
    print(f"Acc {acc*100:.2f}% | Sens {sens*100:.2f}% | Spec {spec*100:.2f}% "
          f"| F1 {f1*100:.2f}% | F1w {f1w*100:.2f}% | BalAcc {bal*100:.2f}%")

    # Log to CSV
    csvp = os.path.join(OUT_DIR, "spectrogram_train_test_results_mobv2_lstm.csv")
    if not os.path.exists(csvp):
        with open(csvp, "w", newline="") as f:
            csv.writer(f).writerow(
                ["Patient","Model","Acc","Sens","Spec","F1","F1w","BalAcc","Thr"]
            )
    with open(csvp, "a", newline="") as f:
        csv.writer(f).writerow([
            PATIENT, "MobileNetV2-LSTM",
            f"{acc*100:.2f}", f"{sens*100:.2f}", f"{spec*100:.2f}",
            f"{f1*100:.2f}", f"{f1w*100:.2f}", f"{bal*100:.2f}", f"{best_th:.2f}"
        ])
    print(f"📁 Results appended to {csvp}")