# =============================================================
# eff_llm_optuna_working.py
# EfficientNet-B0 + LSTM HPO with Optuna + Gemini (REST, retry-safe)
# - Uses your working loader_spectrograms.load_patient_split
# - Uses your working architecture_eff.Net  (EEGSpectrogramEffNetB0_LSTM)
# - Multiple LLM seeds with robust retry (handles 429 / DNS hiccups)
# - Objective prioritizes Accuracy & Sensitivity (equal weight)
# - No dynamic search-space changes (avoids Optuna categorical errors)
# - Windows-friendly (num_workers=0)
# =============================================================
# Run:
#   set SINGLE_PATIENT=chb13              (or PATIENTS=chb01,chb02)
#   set SPEC_DIR=E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed
#   python eff_llm_optuna_working.py
#
# Hard-code API key: set GKEY_HARDCODED below, or use env GEMINI_API_KEY.
# =============================================================

import os, json, time, random, glob, re, csv, requests
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
import optuna
from optuna.samplers import TPESampler

# --- your modules ---
from loader_spectrograms import load_patient_split      # returns ((X_train,y_train),(X_test,y_test))
from architecture_eff import Net                         # EEGSpectrogramEffNetB0_LSTM

# ---------------- config / paths ----------------
SPEC_DIR   = os.environ.get("SPEC_DIR",  r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed")
OUT_ROOT   = os.environ.get("OUT_ROOT",  r"E:\EEG\TRANSFER\models_optuna_eff_gemini")
os.makedirs(OUT_ROOT, exist_ok=True)

# pick one patient quickly via env, or leave blank to auto-discover
SINGLE_PATIENT = os.environ.get("SINGLE_PATIENT", "").strip()  # e.g., "chb13"

# ---------------- torch / device ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "")

# ---------------- search space ----------------
LR_MIN, LR_MAX = 3e-5, 1e-3
BATCH_CHOICES  = [16, 24, 32, 48]
EPOCH_CHOICES  = [60, 90, 120]   # moderate for speed
SEQ_LEN = int(os.environ.get("SEQ_LEN", "3"))

# ---------------- metrics ----------------
CE_WEIGHTS = torch.tensor([1.0, 3.0], device=DEVICE if torch.cuda.is_available() else "cpu")

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc  = (TP+TN)/(TP+TN+FP+FN+1e-8)
    sens = TP/(TP+FN+1e-8)
    spec = TN/(TN+FP+1e-8)
    f1   = 2*TP/(2*TP+FP+FN+1e-8)
    f1w  = f1_score(y_true, y_pred, average='weighted') if cm.size==4 else 0.0
    return acc, sens, spec, f1, f1w

def scan_best_threshold(y_true, y_prob):
    best_val, best_th, best_pack = -1.0, 0.5, None
    for th in np.linspace(0.05, 0.9, 171):
        acc, sens, spec, f1, f1w = compute_metrics(y_true, y_prob, th)
        val = 0.5*acc + 0.5*sens
        if val > best_val:
            best_val, best_th, best_pack = val, th, (acc, sens, spec, f1, f1w)
    return best_th, best_pack, best_val

def oversample_pairs(X, y, factor=4):
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    pos = np.where(y_np == 1)[0]
    neg = np.where(y_np == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    pos_rep = np.tile(pos, factor)
    idx = np.concatenate([neg, pos_rep])
    np.random.shuffle(idx)
    return X[idx], y[idx]

# ---------------- CSV ----------------
CSV_PATH = os.path.join(OUT_ROOT, "effnet_lstm_optuna_gemini_results.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "Patient","Acc","Sens","Spec","F1","F1w","Best_Thr",
            "Trial","lr","batch","epochs","objective"
        ])

# ---------------- Gemini (REST, retry) ----------------
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# !!! Hard-code your Gemini key here OR leave "" to use env GEMINI_API_KEY
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
GKEY_HARDCODED = "AIzaSyAdeKHB0xEO9tb5DJttUPyahqguaI-s3fw"
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

LLM_SEED_CALLS   = int(os.environ.get("LLM_SEED_CALLS", "2"))
MAX_LLM_RETRIES  = int(os.environ.get("MAX_LLM_RETRIES", "6"))  # 0 => infinite retry
INITIAL_BACKOFF  = float(os.environ.get("INITIAL_BACKOFF_SEC", "2.0"))
BACKOFF_MULT     = float(os.environ.get("BACKOFF_MULTIPLIER", "2.0"))
BACKOFF_CAP      = float(os.environ.get("BACKOFF_CAP_SEC", "60.0"))

def _get_gemini_key() -> str:
    key = (GKEY_HARDCODED or os.environ.get("GEMINI_API_KEY","")).strip()
    if not key:
        raise RuntimeError("Gemini key not configured. Set GKEY_HARDCODED or GEMINI_API_KEY.")
    return key

def _gemini_call_with_retry(prompt: str, temperature: float = 0.25) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={_get_gemini_key()}"
    body = {"contents":[{"parts":[{"text":prompt}]}], "generationConfig":{"temperature":temperature}}
    attempt, backoff = 0, INITIAL_BACKOFF
    while True:
        attempt += 1
        try:
            r = requests.post(url, json=body, timeout=90)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait_sec = float(ra) if (ra and ra.isdigit()) else backoff
                print(f"LLM 429 Too Many Requests — sleeping {wait_sec:.1f}s")
                time.sleep(wait_sec)
                backoff = min(backoff * BACKOFF_MULT, BACKOFF_CAP)
                continue
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            if MAX_LLM_RETRIES != 0 and attempt >= MAX_LLM_RETRIES:
                raise RuntimeError(f"Gemini failed after {attempt} attempts: {e}")
            print(f"LLM call failed (attempt {attempt}) — {e}\nRetrying in {backoff:.1f}s ...")
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULT, BACKOFF_CAP)

def _parse_json_block(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(text[s:e+1])

# ---------------- HPO core ----------------
GLOBAL_SEED = int(os.environ.get("SEED", "7"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(GLOBAL_SEED)

N_TRIALS = int(os.environ.get("N_TRIALS", "24"))
SAMPLER  = TPESampler(seed=GLOBAL_SEED, n_startup_trials=min(6, max(3, N_TRIALS//4)))

def run_patient(PATIENT: str):
    print(f"\n================ {PATIENT}: load seq data ================\n")
    (X_train, y_train), (X_test, y_test) = load_patient_split(
        PATIENT, SPEC_DIR, seq_len=SEQ_LEN, test_size=0.3, augment=True
    )
    print(f"{PATIENT}: train {len(y_train)}  test {len(y_test)}")
    print("train labels:", dict(zip(*np.unique(y_train.numpy(), return_counts=True))))
    print("test  labels:", dict(zip(*np.unique(y_test.numpy(), return_counts=True))))

    out_dir = os.path.join(OUT_ROOT, PATIENT); os.makedirs(out_dir, exist_ok=True)
    history: List[Dict[str, Any]] = []

    def build_model():
        return Net(n_classes=2, seq_len=SEQ_LEN, fine_tune=True).to(DEVICE)

    def fit_and_eval(cfg: Dict[str, Any], tag: str) -> float:
        # Oversample positives
        Xb, yb = oversample_pairs(X_train, y_train, factor=4)
        train_loader = DataLoader(
            TensorDataset(Xb, yb),
            batch_size=cfg["batch"],
            shuffle=True,
            num_workers=0,   # Windows safe
            pin_memory=True if DEVICE.type=='cuda' else False,
        )

        model = build_model()
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"]*0.1)
        criterion = nn.CrossEntropyLoss(weight=CE_WEIGHTS.to(DEVICE))

        model.train()
        for _ in range(cfg["epochs"]):
            for Xbt, ybt in train_loader:
                Xbt = Xbt.to(DEVICE, non_blocking=True).float()
                ybt = ybt.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                out = model(Xbt)
                loss = criterion(out, ybt)
                loss.backward(); opt.step()
            sched.step()

        # save
        torch.save(model.state_dict(), os.path.join(out_dir, f"{PATIENT}_{tag}.pt"))

        # eval
        model.eval(); probs=[]
        test_loader = DataLoader(X_test, batch_size=256, shuffle=False, num_workers=0,
                                 pin_memory=True if DEVICE.type=='cuda' else False)
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE, non_blocking=True).float()
                p = torch.softmax(model(data), dim=1)[:,1]
                probs.extend(p.detach().cpu().numpy())
        y_prob = np.array(probs); y_true = y_test.numpy()

        th, pack, obj = scan_best_threshold(y_true, y_prob)
        acc, sens, spec, f1, f1w = pack

        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                PATIENT, f"{acc*100:.2f}", f"{sens*100:.2f}", f"{spec*100:.2f}",
                f"{f1*100:.2f}", f"{f1w*100:.2f}", f"{th:.2f}", tag,
                cfg["lr"], cfg["batch"], cfg["epochs"], f"{obj:.6f}"
            ])

        rec = {"trial": tag, "acc":acc, "sens":sens, "spec":spec, "f1":f1, "thr":th,
               "cfg": cfg, "objective": obj}
        history.append(rec)
        with open(os.path.join(out_dir, f"{PATIENT}_history.json"), "w") as jf:
            json.dump(history, jf, indent=2)
        return obj

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "lr":    trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
            "batch": trial.suggest_categorical("batch", BATCH_CHOICES),
            "epochs":trial.suggest_categorical("epochs", EPOCH_CHOICES),
        }
        return fit_and_eval(cfg, f"optuna{trial.number}")

    study = optuna.create_study(direction="maximize", sampler=SAMPLER, study_name=f"{PATIENT}_eff_study")

    # reasonable seeds
    for lr0 in [3e-5, 7.5e-5, 3e-4]:
        study.enqueue_trial({"lr": lr0, "batch": 32, "epochs": 90})

    # LLM seeds (keep same param names to avoid Optuna dynamic-space issues)
    for i in range(max(0, int(os.environ.get("LLM_SEED_CALLS","2")))):
        try:
            prompt = f'''
Propose ONE hyperparameter set for an EfficientNet-B0 + LSTM EEG spectrogram classifier.
Choose ONLY from:
- lr in log-uniform [{LR_MIN}, {LR_MAX}]
- batch in {BATCH_CHOICES}
- epochs in {EPOCH_CHOICES}
Goal: maximize both accuracy and sensitivity.
Return STRICT JSON: {{"config": {{"lr": float, "batch": int, "epochs": int}}}}
Recent local bests: {json.dumps(sorted(history, key=lambda r: -r.get("objective",0.0))[:5])}
'''
            txt = _gemini_call_with_retry(prompt, temperature=0.25)
            cfg = _parse_json_block(txt)["config"]
            cfg["lr"] = float(min(max(cfg.get("lr", 3e-4), LR_MIN), LR_MAX))
            b = int(cfg.get("batch", 32)); cfg["batch"] = min(BATCH_CHOICES, key=lambda x: abs(x-b))
            ep = int(cfg.get("epochs", 90)); cfg["epochs"] = min(EPOCH_CHOICES, key=lambda x: abs(x-ep))
            study.enqueue_trial({"lr": cfg["lr"], "batch": cfg["batch"], "epochs": cfg["epochs"]})
            print(f"Gemini seed [{i+1}] for {PATIENT}: {cfg}")
        except Exception as e:
            print(f"⚠️ Gemini seed [{i+1}] skipped after retries: {e}")

    # Phase 1
    first_half = max(1, int(os.environ.get("N_TRIALS","24")) // 2)
    print(f"Starting Optuna Phase 1 for {PATIENT} — trials={first_half}")
    study.optimize(objective, n_trials=first_half)

    # Optional mid-run: enqueue one more LLM config (same param names)
    try:
        prompt = f'''
Suggest ONE more promising config inside the SAME spaces:
- lr: [{LR_MIN}, {LR_MAX}] (log-uniform)
- batch: {BATCH_CHOICES}
- epochs: {EPOCH_CHOICES}
Return STRICT JSON: {{"config": {{"lr": float, "batch": int, "epochs": int}}}}
Top results so far: {json.dumps(sorted(history, key=lambda r: -r.get("objective",0.0))[:6])}
'''
        txt = _gemini_call_with_retry(prompt, temperature=0.25)
        cfg = _parse_json_block(txt)["config"]
        cfg["lr"] = float(min(max(cfg.get("lr", 3e-4), LR_MIN), LR_MAX))
        b = int(cfg.get("batch", 32)); cfg["batch"] = min(BATCH_CHOICES, key=lambda x: abs(x-b))
        ep = int(cfg.get("epochs", 90)); cfg["epochs"] = min(EPOCH_CHOICES, key=lambda x: abs(x-ep))
        study.enqueue_trial({"lr": cfg["lr"], "batch": cfg["batch"], "epochs": cfg["epochs"]})
        print(f"Gemini mid-run enqueue for {PATIENT}: {cfg}")
    except Exception as e:
        print(f"⚠️ Gemini mid-run skipped after retries: {e}")

    # Phase 2
    total_trials = int(os.environ.get("N_TRIALS","24"))
    second_half = total_trials - first_half
    if second_half > 0:
        print(f"Starting Optuna Phase 2 for {PATIENT} — trials={second_half}")
        study.optimize(objective, n_trials=second_half)

    # Save best
    best = study.best_trial
    best_cfg = {"lr": best.params["lr"], "batch": best.params["batch"], "epochs": best.params["epochs"]}
    with open(os.path.join(out_dir, f"{PATIENT}_best.json"), "w") as f:
        json.dump({"best_value": best.value, "best_params": best_cfg}, f, indent=2)
    print(f"🏆 Best {PATIENT}: {best_cfg}  objective={best.value:.6f}")
    print("Done:", CSV_PATH)

# ---------------- patients resolution ----------------
def resolve_patients():
    if SINGLE_PATIENT:
        return [SINGLE_PATIENT]
    env_patients = os.environ.get("PATIENTS", "").strip()
    if env_patients:
        return [p.strip() for p in env_patients.split(",") if p.strip()]
    cands = set()
    for f in glob.glob(os.path.join(SPEC_DIR, "*_spectrogram.npy")):
        base = os.path.basename(f)
        m = re.match(r"(chb\d+)_.*_spectrogram\.npy", base)
        if m: cands.add(m.group(1))
    pts = sorted(list(cands))
    if not pts:
        raise SystemExit("No patients found. Set PATIENTS/SINGLE_PATIENT or check SPEC_DIR.")
    return pts

if __name__ == "__main__":
    # Optuna seed & sampler init lives above
    for P in resolve_patients():
        run_patient(P)
