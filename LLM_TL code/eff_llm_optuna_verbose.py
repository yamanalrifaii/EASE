# =============================================================
# eff_llm_optuna_verbose.py
# EfficientNet-B0 + LSTM HPO with Optuna + Gemini (REST, retry)
# Verbose logging: trial start/end, per-epoch loss, LR, timing, GPU mem.
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

# ===== Windows multiprocessing + fast DataLoader config =====
import multiprocessing as mp
torch.multiprocessing.set_sharing_strategy("file_system")

# ---- your modules (match your working pair) ----
from loader_spectrograms import load_patient_split
from architecture_eff import Net  # EEGSpectrogramEffNetB0_LSTM

# ---------------- paths / patient selection ----------------
SPEC_DIR   = os.environ.get("SPEC_DIR",  r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed")
OUT_ROOT   = os.environ.get("OUT_ROOT",  r"E:\EEG\TRANSFER\models_optuna_eff_gemini_verbose")
os.makedirs(OUT_ROOT, exist_ok=True)

# 1) Set here to force a single patient:
SINGLE_PATIENT = os.environ.get("SINGLE_PATIENT", "").strip()  # e.g., "chb13"

# ---------------- device info ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else ''}")
if DEVICE.type == "cuda":
    print(f"[INIT] CUDA: {torch.version.cuda} | torch={torch.__version__}")

# ---------------- search space ----------------
LR_MIN, LR_MAX = 1e-6, 3e-3
BATCH_CHOICES = [32, 64, 80, 96]
EPOCH_CHOICES  = [60, 90, 100, 120]
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
        val = 0.5*acc + 0.5*sens  # equal weight accuracy + sensitivity
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

# ---------------- Gemini (REST with retry/backoff) ----------------
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Hard-code your Gemini key here OR leave "" to use env GEMINI_API_KEY.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
GKEY_HARDCODED = "AIzaSyAdeKHB0xEO9tb5DJttUPyahqguaI-s3fw"
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

LLM_SEED_CALLS  = int(os.environ.get("LLM_SEED_CALLS", "2"))
MAX_LLM_RETRIES = int(os.environ.get("MAX_LLM_RETRIES", "6"))
INITIAL_BACKOFF = float(os.environ.get("INITIAL_BACKOFF_SEC", "2.0"))
BACKOFF_MULT    = float(os.environ.get("BACKOFF_MULTIPLIER", "2.0"))
BACKOFF_CAP     = float(os.environ.get("BACKOFF_CAP_SEC", "60.0"))

def _get_gemini_key() -> str:
    key = (GKEY_HARDCODED or os.environ.get("GEMINI_API_KEY","")).strip()
    if not key or key == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("Gemini key not configured. Set GKEY_HARDCODED or GEMINI_API_KEY.")
    return key

def _gemini_call_with_retry(prompt: str, temperature: float = 0.25) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={_get_gemini_key()}"
    body = {"contents":[{"parts":[{"text":prompt}]}], "generationConfig":{"temperature":temperature}}
    attempt, backoff = 0, INITIAL_BACKOFF
    while True:
        attempt += 1
        try:
            print(f"[LLM] POST (attempt {attempt}) ...")
            r = requests.post(url, json=body, timeout=90)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait_sec = float(ra) if (ra and ra.isdigit()) else backoff
                print(f"[LLM] 429 — sleeping {wait_sec:.1f}s")
                time.sleep(wait_sec)
                backoff = min(backoff * BACKOFF_MULT, BACKOFF_CAP)
                continue
            r.raise_for_status()
            data = r.json()
            txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            print("[LLM] ✓ OK")
            return txt
        except Exception as e:
            if MAX_LLM_RETRIES != 0 and attempt >= MAX_LLM_RETRIES:
                print(f"[LLM] ✗ Failed after {attempt} attempts")
                raise RuntimeError(f"Gemini failed after {attempt} attempts: {e}")
            print(f"[LLM] Error: {e} | retry in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULT, BACKOFF_CAP)

def _parse_json_block(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(text[s:e+1])

# ---------------- Optuna core ----------------
GLOBAL_SEED = int(os.environ.get("SEED", "7"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(GLOBAL_SEED)

N_TRIALS = int(os.environ.get("N_TRIALS", "12"))
SAMPLER  = TPESampler(seed=GLOBAL_SEED, n_startup_trials=min(6, max(3, N_TRIALS//4)))

def run_patient(PATIENT: str):

    print(f"\n========== [{PATIENT}] Loading sequence data ==========")
    t0 = time.time()
    (X_train, y_train), (X_test, y_test) = load_patient_split(
        PATIENT, SPEC_DIR, seq_len=SEQ_LEN, test_size=0.3, augment=True
    )
    print(f"[{PATIENT}] train={len(y_train)} | test={len(y_test)} | load={time.time()-t0:.2f}s")
    print(f"[{PATIENT}] labels train={dict(zip(*np.unique(y_train.numpy(), return_counts=True)))} "
          f"test={dict(zip(*np.unique(y_test.numpy(), return_counts=True)))}")

    out_dir = os.path.join(OUT_ROOT, PATIENT); os.makedirs(out_dir, exist_ok=True)
    history: List[Dict[str, Any]] = []

    def build_model():
        model = Net(n_classes=2, seq_len=SEQ_LEN, fine_tune=True).to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            print(f"[{PATIENT}] Model on CUDA; reset peak mem")
        return model

    def fit_and_eval(cfg: Dict[str, Any], tag: str) -> float:
        print(f"\n--- [{PATIENT}] Trial {tag} START ---")
        print(f"cfg={cfg}")
        t_trial = time.time()

        # Oversample positives
        Xb, yb = oversample_pairs(X_train, y_train, factor=4)
        train_loader = DataLoader(
            TensorDataset(Xb, yb),
            batch_size=cfg["batch"],
            shuffle=True,
            num_workers=6,
            persistent_workers=True,    # ✔ BIG speed boost
            pin_memory=False,           # ✔ Windows must have this OFF
            prefetch_factor=4,
        )

        model = build_model()
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"]*0.1)
        criterion = nn.CrossEntropyLoss(weight=CE_WEIGHTS.to(DEVICE))


        # Train
        e0 = time.time()
        for e in range(1, cfg["epochs"]+1):
            model.train(); running_loss=0.0; n_batches=0
            t_epoch = time.time()
            for Xbt, ybt in train_loader:
                Xbt = Xbt.to(DEVICE, non_blocking=True).float()
                ybt = ybt.to(DEVICE, non_blocking=True)                
                # ---------- MIXUP AUGMENTATION ----------
                if random.random() < 0.15:  # 15% of batches
                    lam = np.random.beta(0.2, 0.2)
                    index = torch.randperm(Xbt.size(0)).to(DEVICE)

                    Xbt_mix = lam * Xbt + (1 - lam) * Xbt[index]
                    ybt_shuffled = ybt[index]

                    mixup_active = True
                else:
                    Xbt_mix = Xbt
                    mixup_active = False
                # -----------------------------------------

                opt.zero_grad()
                out = model(Xbt_mix)

                # Use mixed loss only when MixUp was applied
                if mixup_active:
                    loss = lam * criterion(out, ybt) + (1 - lam) * criterion(out, ybt_shuffled)
                else:
                    loss = criterion(out, ybt)

                loss.backward(); opt.step()
                running_loss += loss.detach().item()
                n_batches += 1
                
            sched.step()
            avg_loss = running_loss/max(1,n_batches)
            cur_lr = sched.get_last_lr()[0]
            elapsed = time.time()-t_epoch
            if e % max(1, cfg["epochs"]//10) == 0 or e == 1 or e == cfg["epochs"]:
                gpu_mem = ""
                if DEVICE.type == "cuda":
                    peak = torch.cuda.max_memory_allocated()/1024/1024
                    gpu_mem = f" | peak_gpu_mem={peak:.1f}MB"
                print(f"[{PATIENT}] Epoch {e:03d}/{cfg['epochs']} | loss={avg_loss:.5f} | lr={cur_lr:.2e} | {elapsed:.1f}s{gpu_mem}")

        train_time = time.time()-e0
        print(f"[{PATIENT}] Train time = {train_time/60:.2f} min")

        # Save model
        model_path = os.path.join(out_dir, f"{PATIENT}_{tag}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"[{PATIENT}] ✓ Saved: {model_path}")
        
        torch.cuda.empty_cache()

        # Evaluate
        model.eval(); probs=[]
        test_loader = DataLoader(
            X_test,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=False,
            prefetch_factor=4,
)
        t_eval = time.time()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE, non_blocking=True).float()
                p = torch.softmax(model(data), dim=1)[:,1]
                probs.extend(p.detach().cpu().numpy())
        y_prob = np.array(probs); y_true = y_test.numpy()
        th, pack, obj = scan_best_threshold(y_true, y_prob)
        acc, sens, spec, f1, f1w = pack
        print(f"[{PATIENT}] Eval {time.time()-t_eval:.2f}s | best_thr={th:.2f}")
        print(f"[{PATIENT}] Acc={acc*100:.2f}% Sens={sens*100:.2f}% Spec={spec*100:.2f}% "
              f"F1={f1*100:.2f}% F1w={f1w*100:.2f}% | Obj(0.5A+0.5S)={obj:.6f}")

        # CSV + history
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

        print(f"--- [{PATIENT}] Trial {tag} END | wall={time.time()-t_trial:.1f}s ---\n")
        torch.cuda.empty_cache()        
        return obj

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "lr":    trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
            "batch": trial.suggest_categorical("batch", BATCH_CHOICES),
            "epochs":trial.suggest_categorical("epochs", EPOCH_CHOICES),
        }
        print(f"[Optuna] -> START trial#{trial.number} cfg={cfg}")
        val = fit_and_eval(cfg, f"optuna{trial.number}")
        print(f"[Optuna] <- END   trial#{trial.number} value={val:.6f}")
        return val

    study = optuna.create_study(direction="maximize", sampler=SAMPLER, study_name=f"{PATIENT}_eff_study")

    # Manual seeds (enqueue)
    for lr0 in [3e-5, 7.5e-5, 3e-4]:
        study.enqueue_trial({"lr": lr0, "batch": 32, "epochs": 90})

    # LLM seeds
    for i in range(max(0, LLM_SEED_CALLS)):
        try:
            prompt = f"""
Propose ONE hyperparameter set for an EfficientNet-B0 + LSTM EEG spectrogram classifier.
Choose ONLY from:
- lr in log-uniform [{LR_MIN}, {LR_MAX}]
- batch in {BATCH_CHOICES}
- epochs in {EPOCH_CHOICES}
Goal: maximize both accuracy and sensitivity.
Return STRICT JSON: {{"config": {{"lr": float, "batch": int, "epochs": int}}}}
Recent local bests: {json.dumps(sorted(history, key=lambda r: -r.get('objective',0.0))[:5])}
"""
            txt = _gemini_call_with_retry(prompt, temperature=0.25)
            cfg = _parse_json_block(txt)["config"]
            cfg["lr"] = float(min(max(cfg.get("lr", 3e-4), LR_MIN), LR_MAX))
            b = int(cfg.get("batch", 32)); cfg["batch"] = min(BATCH_CHOICES, key=lambda x: abs(x-b))
            ep = int(cfg.get("epochs", 90)); cfg["epochs"] = min(EPOCH_CHOICES, key=lambda x: abs(x-ep))
            study.enqueue_trial({"lr": cfg["lr"], "batch": cfg["batch"], "epochs": cfg["epochs"]})
            print(f"[LLM] Seed {i+1}: {cfg}")
        except Exception as e:
            print(f"[LLM] Seed {i+1} skipped: {e}")

    # Phase 1
    total_trials = int(os.environ.get("N_TRIALS","12"))
    first_half = max(1, total_trials // 2)
    print(f"[Optuna] Phase-1 n_trials={first_half}")
    study.optimize(objective, n_trials=first_half)

    # Optional mid-run enqueue (same space)
    try:
        prompt = f"""
Suggest ONE more promising config inside the SAME spaces:
- lr: [{LR_MIN}, {LR_MAX}] (log-uniform)
- batch: {BATCH_CHOICES}
- epochs: {EPOCH_CHOICES}
Return STRICT JSON: {{"config": {{"lr": float, "batch": int, "epochs": int}}}}
Top results so far: {json.dumps(sorted(history, key=lambda r: -r.get('objective',0.0))[:6])}
"""
        txt = _gemini_call_with_retry(prompt, temperature=0.25)
        cfg = _parse_json_block(txt)["config"]
        cfg["lr"] = float(min(max(cfg.get("lr", 3e-4), LR_MIN), LR_MAX))
        b = int(cfg.get("batch", 64)); cfg["batch"] = min(BATCH_CHOICES, key=lambda x: abs(x-b))
        ep = int(cfg.get("epochs", 90)); cfg["epochs"] = min(EPOCH_CHOICES, key=lambda x: abs(x-ep))
        study.enqueue_trial({"lr": cfg["lr"], "batch": cfg["batch"], "epochs": cfg["epochs"]})
        print(f"[LLM] Mid-run enqueue: {cfg}")
    except Exception as e:
        print(f"[LLM] Mid-run skipped: {e}")

    # Phase 2
    second_half = total_trials - first_half
    if second_half > 0:
        print(f"[Optuna] Phase-2 n_trials={second_half}")
        study.optimize(objective, n_trials=second_half)

    # Save best
    best = study.best_trial
    best_cfg = {"lr": best.params["lr"], "batch": best.params["batch"], "epochs": best.params["epochs"]}
    with open(os.path.join(out_dir, f"{PATIENT}_best.json"), "w") as f:
        json.dump({"best_value": best.value, "best_params": best_cfg}, f, indent=2)
    print(f"[RESULT] Best {PATIENT}: {best_cfg}  objective={best.value:.6f}")
    print(f"[RESULT] CSV: {CSV_PATH}")

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
    mp.freeze_support()
    mp.set_start_method('spawn')
    for P in resolve_patients():
        run_patient(P)
