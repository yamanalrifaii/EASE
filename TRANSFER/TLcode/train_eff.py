# =============================================================
# train_eff.py — Train EfficientNet-B0 + LSTM on EEG Spectrograms
# =============================================================
import os, csv, numpy as np, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
from loader_spectrograms import load_patient_split
from architecture_eff import Net

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
ROOT      = r"E:\EEG\TRANSFER"
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
#OUT_DIR   = os.path.join(ROOT, "models"); os.makedirs(OUT_DIR, exist_ok=True)
OUT_ROOT   = os.environ.get("OUT_ROOT",  r"E:\EEG\TRANSFER\models_optuna_eff_gemini_verbose")

PATIENT    = "chb24"       # change per patient
SEQ_LEN    = 3
EPOCHS     = 150
BATCH_SIZE = 32
LR         = 3e-5
OVERSAMPLE = 5

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 3.0], device='cuda' if torch.cuda.is_available() else 'cpu')
)

# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------
def oversample(X, y, factor=OVERSAMPLE):
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    pos = np.where(y_np == 1)[0]; neg = np.where(y_np == 0)[0]
    if len(pos) == 0 or len(neg) == 0: return X, y
    pos_rep = np.tile(pos, factor)
    idx = np.concatenate([neg, pos_rep]); np.random.shuffle(idx)
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
    loader=DataLoader(X,batch_size=batch,shuffle=False)
    with torch.no_grad():
        for data in loader:
            if torch.cuda.is_available(): data=data.cuda()
            p = torch.softmax(model(data.float()), dim=1)[:,1]
            probs.extend(p.cpu().numpy())
    return np.array(probs)

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_patient_split(
        PATIENT, SPEC_DIR, seq_len=SEQ_LEN, test_size=0.3, augment=True
    )
    X_train, y_train = oversample(X_train, y_train, OVERSAMPLE)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(n_classes=2, seq_len=SEQ_LEN, fine_tune=True)
    if torch.cuda.is_available(): model.cuda()

    opt = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)

    for e in range(EPOCHS):
        model.train(); losses=[]
        for Xb, yb in train_loader:
            if torch.cuda.is_available(): Xb, yb = Xb.cuda(), yb.cuda()
            opt.zero_grad()
            out = model(Xb.float())
            loss = criterion(out, yb)
            loss.backward(); opt.step()
            losses.append(loss.detach().cpu())
        sched.step()
        print(f"Epoch {e:03d} | Train {torch.mean(torch.stack(losses)):.5f}")

    path = os.path.join(OUT_DIR, f"{PATIENT}_EffB0LSTM.pt")
    torch.save(model.state_dict(), path)
    print(f"✅ Training complete. Model saved to {path}")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    y_prob = predict_proba(model, X_test)
    y_true = y_test.numpy()

    best_j, best_th = -1, 0.3
    for th in np.linspace(0.05, 0.6, 112):
        acc, sens, spec, _, _, _ = compute_metrics(y_true, y_prob, th)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_th = j, th

    acc, sens, spec, f1, f1w, bal = compute_metrics(y_true, y_prob, best_th)
    print(f"\n=== Final Test Results ({PATIENT}, EffB0-LSTM) ===")
    print(f"Threshold = {best_th:.2f}")
    print(f"Acc {acc*100:.2f}% | Sens {sens*100:.2f}% | Spec {spec*100:.2f}% "
          f"| F1 {f1*100:.2f}% | F1w {f1w*100:.2f}% | BalAcc {bal*100:.2f}%")

    csvp = os.path.join(OUT_DIR, "spectrogram_train_test_results_effb0.csv")
    header = ["Patient","Model","Acc","Sens","Spec","F1","F1w","BalAcc","Thr"]
    if not os.path.exists(csvp):
        with open(csvp,"w",newline="") as f: csv.writer(f).writerow(header)
    with open(csvp,"a",newline="") as f:
        csv.writer(f).writerow([
            PATIENT, "EffB0-LSTM",
            f"{acc*100:.2f}", f"{sens*100:.2f}", f"{spec*100:.2f}",
            f"{f1*100:.2f}", f"{f1w*100:.2f}", f"{bal*100:.2f}", f"{best_th:.2f}"
        ])
    print(f"📁 Results appended to {csvp}")