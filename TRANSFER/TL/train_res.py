# =============================================================
# train_resnet34_cnn.py — ResNet-34 CNN Training + Fine-tuning
# =============================================================
import os, numpy as np, torch, csv
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from loader_spectrograms import load_patient_split
from architecture_res import Net
from trainer import trainer

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
ROOT      = r"E:\EEG\TRANSFER"
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
OUT_DIR   = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENT    = "chb02"      # change per run
EPOCHS     = 100          # stage 1 epochs
FT_EPOCHS  = 25           # fine-tuning epochs
BATCH_SIZE = 64
LR         = 1e-4         # base learning rate
FT_LR      = 3e-5         # fine-tuning learning rate

# -------------------------------------------------------------
# METRICS + HELPERS
# -------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN = FP = FN = TP = 0
    acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
    sens = TP / (TP + FN) if (TP + FN) else 0
    spec = TN / (TN + FP) if (TN + FP) else 0
    f1b  = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0
    f1w  = f1_score(y_true, y_pred, average='weighted') if cm.size == 4 else 0
    bal_acc = (sens + spec) / 2
    return acc, sens, spec, f1b, f1w, bal_acc

def predict_proba(model, X):
    """Return class-1 probabilities for tensor dataset."""
    model.eval()
    probs = []
    loader = DataLoader(X, batch_size=512, shuffle=False)
    with torch.no_grad():
        for data in loader:
            if torch.cuda.is_available():
                data = data.cuda()
            data = data.squeeze(1)             # remove seq dim
            logits = model(data.float())
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
    return np.array(probs)

# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------
if __name__ == "__main__":
    # === Load Data ===
    Train_set, Test_set = load_patient_split(PATIENT, SPEC_DIR, seq_len=1, augment=True)
    X_train, y_train = Train_set
    X_test,  y_test  = Test_set

    # === Stage 1: Train top layers ===
    model = Net(n_classes=2)
    t = trainer(model, Train_set)
    t.compile(lr=LR)

    out_path = os.path.join(OUT_DIR, f"{PATIENT}_ResNet34CNN_best.pt")

    print(f"\n================ Stage 1 Training ({PATIENT}) ================")
    t.train(epochs=EPOCHS, batch_size=BATCH_SIZE, directory=out_path)

    # === Stage 2: Fine-tune entire model ===
    print("\n🔧 Starting fine-tuning (all layers trainable)...")
    for p in model.parameters():
        p.requires_grad = True

    t = trainer(model, Train_set)
    t.compile(lr=FT_LR)
    t.train(epochs=FT_EPOCHS, batch_size=BATCH_SIZE, directory=out_path)
    print("✅ Fine-tuning complete.")

    # === Evaluation ===
    model.load_state_dict(torch.load(out_path, map_location="cpu"))
    if torch.cuda.is_available():
        model.cuda()
    y_prob_test = predict_proba(model, X_test)

    # Optimal threshold (Youden’s J)
    best_j, best_th = -1, 0.5
    for th in np.linspace(0.2, 0.6, 41):
        acc, sens, spec, _, _, _ = compute_metrics(y_test.numpy(), y_prob_test, threshold=th)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_th = j, th

    acc, sens, spec, f1, f1w, bal_acc = compute_metrics(y_test.numpy(), y_prob_test, threshold=best_th)
    print(f"\n=== Final Test Results ({PATIENT}, ResNet-34 CNN) ===")
    print(f"Optimal threshold (Youden’s J) = {best_th:.2f}")
    print(f"Acc {acc*100:.2f}% | Sens {sens*100:.2f}% | Spec {spec*100:.2f}% | "
          f"F1 {f1*100:.2f}% | F1w {f1w*100:.2f}% | Balanced Acc {bal_acc*100:.2f}%")
    print(f"✅ Model saved to {out_path}")

    # === Log results ===
    csv_path = os.path.join(OUT_DIR, "spectrogram_train_test_results.csv")
    header = ["Patient","Set","Model","Accuracy","Sensitivity","Specificity",
              "F1","F1_weighted","Balanced_Accuracy","Threshold"]
    if not os.path.exists(csv_path):
        with open(csv_path,"w",newline="") as f: csv.writer(f).writerow(header)
    with open(csv_path,"a",newline="") as f:
        csv.writer(f).writerow([
            PATIENT,"Test","ResNet34_CNN",
            f"{acc*100:.2f}",f"{sens*100:.2f}",f"{spec*100:.2f}",
            f"{f1*100:.2f}",f"{f1w*100:.2f}",f"{bal_acc*100:.2f}",f"{best_th:.2f}"
        ])
    print(f"📁 Results appended to {csv_path}")