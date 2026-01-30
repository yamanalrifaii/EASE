# =============================================================
# train_eff_all_stream.py — Train EfficientNet-B0 + LSTM (Streaming)
# =============================================================
import os, csv, numpy as np, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, f1_score

# ⬇️ new streaming dataset
from loader_all import EEGSpectrogramStreamDataset
from architecture_eff import Net

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
ROOT      = r"E:\EEG\TRANSFER"
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
OUT_DIR   = os.path.join(ROOT, "models_all"); os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN     = 3
EPOCHS      = 20          # we will break early in debug
BATCH_SIZE  = 8           # small batch fits in RAM
LR          = 3e-5

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 3.0], device='cuda' if torch.cuda.is_available() else 'cpu')
)

# -------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Build streaming dataset (never loads all into memory)
    full_ds = EEGSpectrogramStreamDataset(SPEC_DIR, seq_len=SEQ_LEN, augment=True)
    train_len = int(0.7 * len(full_ds))
    test_len  = len(full_ds) - train_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2️⃣ Model
    model = Net(n_classes=2, seq_len=SEQ_LEN, fine_tune=True)
    if torch.cuda.is_available(): 
        model.cuda()

    opt = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)

    # ---------------------------------------------------------
    # 3️⃣ Training loop (with debug prints)
    # ---------------------------------------------------------
    print("✅ Starting training loop...")

    for e in range(EPOCHS):
        print(f"\n--- Epoch {e} starting ---")
        model.train()
        losses = []

        # step 1: test if dataloader works
        try:
            first_batch = next(iter(train_loader))
            print(f"✅ First batch loaded successfully! {len(first_batch[0])} samples per batch.")
        except Exception as err:
            print("❌ ERROR while loading first batch:", err)
            raise

        # real loop
        for batch_idx, (Xb, yb) in enumerate(train_loader):
            print(f"  🔹 Batch {batch_idx} loaded. Shape = {tuple(Xb.shape)}")
            try:
                if torch.cuda.is_available():
                    print("  ⚙️ Moving to GPU...")
                    Xb = Xb.cuda(non_blocking=True)
                    yb = yb.cuda(non_blocking=True)

                opt.zero_grad()
                print("  🧮 Forward pass...")
                out = model(Xb)
                print("  🔁 Backward + step...")
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                losses.append(loss.detach().cpu())

            except Exception as err:
                print(f"❌ ERROR in batch {batch_idx}:", err)
                raise

            # only first 3 batches for debug
            # if batch_idx >= 2:
                # print("🧩 Stopping early (debug mode after 3 batches).")
                # break

        sched.step()
        if len(losses) > 0:
            print(f"✅ Epoch {e:03d} | Avg loss: {torch.mean(torch.stack(losses)):.5f}")
        else:
            print(f"⚠️ Epoch {e:03d} | No batches processed.")

        print(f"--- Epoch {e} done ---")

        # stop after first epoch during debug
        # break

    # ---------------------------------------------------------
    # 4️⃣ Save final model (even debug-trained)
    # ---------------------------------------------------------
    path = os.path.join(OUT_DIR, "AllPatients_EffB0LSTM_stream_DEBUG.pt")
    torch.save(model.state_dict(), path)
    print(f"✅ Training complete (debug). Model saved to {path}")

    # ---------------------------------------------------------
    # 5️⃣ Evaluation (stream test set) — optional in debug
    # ---------------------------------------------------------
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            if torch.cuda.is_available():
                Xb = Xb.cuda()
            p = torch.softmax(model(Xb.float()), dim=1)[:,1]
            probs.extend(p.cpu().numpy())
            labels.extend(yb.numpy())

    y_prob = np.array(probs)
    y_true = np.array(labels)

    best_j, best_th = -1, 0.3
    for th in np.linspace(0.05, 0.6, 112):
        acc, sens, spec, _, _, _ = compute_metrics(y_true, y_prob, th)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_th = j, th

    acc, sens, spec, f1, f1w, bal = compute_metrics(y_true, y_prob, best_th)
    print(f"\n=== Final Test Results (All Patients, EffB0-LSTM STREAM DEBUG) ===")
    print(f"Threshold = {best_th:.2f}")
    print(f"Acc {acc*100:.2f}% | Sens {sens*100:.2f}% | Spec {spec*100:.2f}% "
          f"| F1 {f1*100:.2f}% | F1w {f1w*100:.2f}% | BalAcc {bal*100:.2f}%")