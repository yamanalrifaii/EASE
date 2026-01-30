# =============================================================
# eff_llm_optuna_fast_win.py
# EfficientNet-B0 + LSTM for EEG Spectrograms (Windows-friendly)
#   • Optuna (TPE) HPO with MedianPruner
#   • Gemini LLM seeding + mid-run refinement (multi-calls, retry)
#   • AMP + pinned memory + non_blocking H2D copies
#   • Windows-safe DataLoader defaults (num_workers=0)
#   • Memory-safe loader via loader_spectrograms_win (float16 + mmap)
#   • Objective balances Accuracy & Sensitivity equally
# Requires: pip install optuna requests
# Run example (cmd):
#   set PATIENTS=chb13
#   python eff_llm_optuna_fast_win.py
# =============================================================

import os, json, time, random, csv, re, glob, requests, gc
from typing import Dict, Any, List, Tuple
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ====== Your modules ======
from loader_spectrograms_win import load_patient_split      # ((X_train,y_train),(X_test,y_test))
from architecture_eff import Net                            # your model (EfficientNet-B0 + LSTM, etc.)

# =========================
# PATHS / PATIENTS
# =========================
SPEC_DIR   = os.environ.get("SPEC_DIR",  r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed")
OUT_ROOT   = os.environ.get("OUT_ROOT",  r"E:\EEG\TRANSFER\models_optuna_eff_gemini")
os.makedirs(OUT_ROOT, exist_ok=True)

_env = os.environ.get("PATIENTS", "").strip()
if _env:
    PATIENTS = [p.strip() for p in _env.split(",") if p.strip()]
else:
    cands = set()
    for f in glob.glob(os.path.join(SPEC_DIR, "*_spectrogram.npy")):
        base = os.path.basename(f)
        m = re.match(r"(chb\d+)_.*_spectrogram\.npy", base)
        if m: cands.add(m.group(1))
    PATIENTS = sorted(list(cands))
if not PATIENTS:
    raise SystemExit("No patients found. Set PATIENTS or check SPEC_DIR.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Running on: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")

# ---- Windows-safe DataLoader defaults ----
IS_WIN = (os.name == "nt")
DL_WORKERS = int(os.environ.get("DL_WORKERS", "0" if IS_WIN else "2"))
PIN_MEM    = bool(int(os.environ.get("PIN_MEMORY", "1"))) and (torch.cuda.is_available())

# =========================
# GEMINI (multiple calls + retry)
# =========================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# !!! IMPORTANT: Hard-code your key here OR leave empty and set env GEMINI_API_KEY
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
GKEY_HARDCODED = "AIzaSyAdeKHB0xEO9tb5DJttUPyahqguaI-s3fw"
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# LLM call budgets
LLM_SEED_CALLS   = int(os.environ.get("LLM_SEED_CALLS", "2"))    # before Phase-1
LLM_REFINE_CALLS = int(os.environ.get("LLM_REFINE_CALLS", "1"))  # between phases

# Retry behavior
MAX_LLM_RETRIES      = int(os.environ.get("MAX_LLM_RETRIES", "6"))  # 0 => infinite retry
INITIAL_BACKOFF_SEC  = float(os.environ.get("INITIAL_BACKOFF_SEC", "2.0"))
BACKOFF_MULTIPLIER   = float(os.environ.get("BACKOFF_MULTIPLIER", "2.0"))
BACKOFF_CAP_SEC      = float(os.environ.get("BACKOFF_CAP_SEC", "60.0"))

def _resolve_gemini_key() -> str:
    if GKEY_HARDCODED and GKEY_HARDCODED.strip() and GKEY_HARDCODED != "YOUR_GEMINI_API_KEY_HERE":
        return GKEY_HARDCODED.strip()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Gemini key not configured. Set GKEY_HARDCODED or GEMINI_API_KEY.")
    return key

GEMINI_KEY = _resolve_gemini_key()

def _gemini_call_with_retry(prompt: str, temperature: float = 0.25) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    body = {"contents":[{"parts":[{"text":prompt}]}], "generationConfig":{"temperature":temperature}}
    attempt = 0
    backoff = INITIAL_BACKOFF_SEC
    while True:
        attempt += 1
        try:
            r = requests.post(url, json=body, timeout=90)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait_sec = float(ra) if ra and ra.isdigit() else backoff
                print(f"LLM 429 Too Many Requests — sleeping {wait_sec:.1f}s")
                time.sleep(wait_sec)
                backoff = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_CAP_SEC)
                continue
            r.raise_for_status()
            data = r.json()
            txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return txt
        except Exception as e:
            if (MAX_LLM_RETRIES != 0) and (attempt >= MAX_LLM_RETRIES):
                raise RuntimeError(f"Gemini failed after {attempt} attempts: {e}")
            print(f"LLM call failed (attempt {attempt}) — {e}\nRetrying in {backoff:.1f}s ...")
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_CAP_SEC)

def _parse_strict_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(text[s:e+1])

# =========================
# SEARCH SPACE
# =========================
LR_MIN, LR_MAX = 3e-5, 1e-3
BATCH_CHOICES  = [16, 24, 32]
EPOCH_CHOICES  = [40, 60, 90]

# =========================
# METRICS
# =========================
CE_WEIGHTS = torch.tensor([1.0, 3.0], device=DEVICE if torch.cuda.is_available() else "cpu")

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc  = (TP+TN)/(TP+TN+FP+FN+1e-8)
    sens = TP/(TP+FN+1e-8)
    spec = TN/(TN+FP+1e-8)
    f1   = 2*TP/(2*TP+FP+FN+1e-8)
    f1w  = f1_score(y_true, y_pred, average='weighted') if cm.size==4 else 0
    return acc, sens, spec, f1, f1w

def scan_best_threshold(y_true, y_prob):
    best_val, best_th, best_pack = -1, 0.5, None
    for th in np.linspace(0.05, 0.9, 171):
        acc, sens, spec, f1, f1w = compute_metrics(y_true, y_prob, th)
        score = 0.5*acc + 0.5*sens
        if score > best_val:
            best_val, best_th, best_pack = score, th, (acc, sens, spec, f1, f1w)
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

# =========================
# CSV LOG
# =========================
CSV_PATH = os.path.join(OUT_ROOT, "effnet_lstm_optuna_gemini_results.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "Patient","Acc","Sens","Spec","F1","F1w","Best_Thr",
            "Trial","lr","batch","epochs","objective"
        ])

# =========================
# HPO LOOP
# =========================
GLOBAL_SEED = int(os.environ.get("SEED", "7"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

N_TRIALS = int(os.environ.get("N_TRIALS", "24"))
SAMPLER  = TPESampler(seed=GLOBAL_SEED, n_startup_trials=min(6, max(3, N_TRIALS//4)))
PRUNER   = MedianPruner(n_startup_trials=2, n_warmup_steps=2)

for PATIENT in PATIENTS:
    print(f"\n================ {PATIENT}: load seq data ================")
    (X_train, y_train), (X_test, y_test) = load_patient_split(
        PATIENT, SPEC_DIR, seq_len=3, test_size=0.3, augment=True
    )
    print(f"{PATIENT}: train {len(X_train)}  test {len(X_test)}")
    def _hist(t):
        u,c = np.unique(t.numpy(), return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))
    print("train labels:", _hist(y_train))
    print("test  labels:", _hist(y_test))
    print("Sample batch mean/std:", float(X_train.float().mean()), float(X_train.float().std()))

    out_dir = os.path.join(OUT_ROOT, PATIENT)
    os.makedirs(out_dir, exist_ok=True)

    history: List[Dict[str, Any]] = []

    def build_model():
        model = Net(n_classes=2, seq_len=3, fine_tune=True)
        return model.to(DEVICE)

    def fit_and_eval(cfg: Dict[str, Any], tag: str, trial: optuna.Trial = None) -> float:
        Xb, yb = oversample_pairs(X_train, y_train, factor=4)
        train_loader = DataLoader(
            TensorDataset(Xb, yb),
            batch_size=cfg["batch"], shuffle=True,
            num_workers=DL_WORKERS, pin_memory=PIN_MEM, persistent_workers=(DL_WORKERS>0)
        )

        model = build_model()
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"]*0.1)
        criterion = nn.CrossEntropyLoss(weight=CE_WEIGHTS.to(DEVICE))
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        try:
            model.train()
            for e in range(cfg["epochs"]):
                try:
                    for Xbt, ybt in train_loader:
                        Xbt = Xbt.to(DEVICE, non_blocking=True).float()
                        ybt = ybt.to(DEVICE, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            out = model(Xbt)
                            loss = criterion(out, ybt)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        print("⚠️ Worker crash detected → falling back to num_workers=0")
                        train_loader = DataLoader(
                            TensorDataset(Xb, yb),
                            batch_size=cfg["batch"], shuffle=True,
                            num_workers=0, pin_memory=PIN_MEM, persistent_workers=False
                        )
                        for Xbt, ybt in train_loader:
                            Xbt = Xbt.to(DEVICE, non_blocking=True).float()
                            ybt = ybt.to(DEVICE, non_blocking=True)
                            opt.zero_grad(set_to_none=True)
                            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                                out = model(Xbt)
                                loss = criterion(out, ybt)
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                    else:
                        raise
                sched.step()

                if (e+1) % 10 == 0 or e == 0:
                    print(f"[{tag}] epoch {e+1}/{cfg['epochs']} loss={loss.item():.4f}")
                if trial is not None:
                    proxy = -float(loss.item())
                    trial.report(proxy, step=e)
                    if trial.should_prune():
                        raise optuna.TrialPruned("MedianPruner triggered")

            # save
            pt_path = os.path.join(out_dir, f"{PATIENT}_{tag}.pt")
            torch.save(model.state_dict(), pt_path)

            # eval
            model.eval()
            probs=[]
            with torch.no_grad():
                test_loader = DataLoader(X_test, batch_size=256, shuffle=False,
                                         num_workers=0, pin_memory=PIN_MEM)
                for data in test_loader:
                    data = data.to(DEVICE, non_blocking=True).float()
                    p = torch.softmax(model(data), dim=1)[:,1]
                    probs.extend(p.cpu().numpy())
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

        except KeyboardInterrupt:
            raise optuna.TrialPruned("User interrupted / KeyboardInterrupt")
        finally:
            del model, train_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def objective(trial: optuna.Trial) -> float:
        refined = trial.study.user_attrs.get("refined", False)
        if refined:
            lr_min = float(trial.study.user_attrs.get("lr_min", LR_MIN))
            lr_max = float(trial.study.user_attrs.get("lr_max", LR_MAX))
            batches = list(trial.study.user_attrs.get("batches", BATCH_CHOICES))
            epochs  = list(trial.study.user_attrs.get("epochs", EPOCH_CHOICES))
            cfg = {
                "lr":     trial.suggest_float("lr_r2", lr_min, lr_max, log=True),
                "batch":  trial.suggest_categorical("batch_r2", batches),
                "epochs": trial.suggest_categorical("epochs_r2", epochs),
            }
        else:
            cfg = {
                "lr":     trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
                "batch":  trial.suggest_categorical("batch", BATCH_CHOICES),
                "epochs": trial.suggest_categorical("epochs", EPOCH_CHOICES),
            }
        return fit_and_eval(cfg, f"optuna{trial.number}", trial)

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=random.randint(1, 10**6)),
                                pruner=PRUNER,
                                study_name=f"{PATIENT}_eff_study")

    # seed known-good nearby configs
    for lr0 in [3e-5, 7.5e-5, 3e-4]:
        study.enqueue_trial({"lr": lr0, "batch": 24, "epochs": 60})

    # -------- LLM SEED (multiple calls) --------
    for i in range(max(0, LLM_SEED_CALLS)):
        try:
            prompt = f"""
Propose ONE hyperparameter set for a 2-class EEG spectrogram model (EfficientNet-B0 + LSTM).
Choose ONLY from:
- lr in log-uniform [{LR_MIN}, {LR_MAX}]
- batch in {BATCH_CHOICES}
- epochs in {EPOCH_CHOICES}
Optimize equally for accuracy and sensitivity.
Return STRICT JSON: {{"config": {{"lr": float, "batch": int, "epochs": int}}}}
Recent local bests: {json.dumps(sorted(history, key=lambda r: -r['objective'])[:5])}
"""
            txt = _gemini_call_with_retry(prompt, temperature=0.25)
            cfg = _parse_strict_json(txt)["config"]
            cfg["lr"]    = float(min(max(cfg.get("lr", 3e-4), LR_MIN), LR_MAX))
            b = int(cfg.get("batch", 24)); cfg["batch"] = min(BATCH_CHOICES, key=lambda x: abs(x-b))
            ep = int(cfg.get("epochs", 60)); cfg["epochs"] = min(EPOCH_CHOICES, key=lambda x: abs(x-ep))
            study.enqueue_trial(cfg)
            print(f"Gemini seed [{i+1}/{LLM_SEED_CALLS}] for {PATIENT}: {cfg}")
        except Exception as e:
            print(f"⚠️ Gemini seed [{i+1}] skipped after retries: {e}")

    # ---- Phase 1 ----
    first_half = max(1, N_TRIALS // 2)
    print(f"Starting Optuna Phase 1 for {PATIENT} — trials={first_half}")
    study.optimize(objective, n_trials=first_half)

    # -------- LLM REFINE (multiple calls) --------
    for j in range(max(0, LLM_REFINE_CALLS)):
        try:
            top = sorted(history, key=lambda r: (-r["objective"], -r["sens"]))[:6]
            prompt = f"""
Refine search ranges using top results then propose ONE promising config.
Original:
- lr: [{LR_MIN}, {LR_MAX}] (log-uniform)
- batch: {BATCH_CHOICES}
- epochs: {EPOCH_CHOICES}
Top results: {json.dumps(top)}
Return STRICT JSON:
{{
 "refined": {{"lr_min": float, "lr_max": float, "batches": {BATCH_CHOICES}, "epochs": {EPOCH_CHOICES}}},
 "config":  {{"lr": float, "batch": int, "epochs": int}}
}}
Constraints: lr_min>= {LR_MIN}, lr_max<= {LR_MAX}, lr_min<lr_max; batches/epochs must be non-empty subsets.
"""
            txt = _gemini_call_with_retry(prompt, temperature=0.25)
            data = _parse_strict_json(txt)
            ref = data.get("refined", {})
            lr_min = float(max(LR_MIN, ref.get("lr_min", LR_MIN)))
            lr_max = float(min(LR_MAX, ref.get("lr_max", LR_MAX)))
            if lr_min >= lr_max: lr_min, lr_max = LR_MIN, LR_MAX
            batches = [b for b in ref.get("batches", BATCH_CHOICES) if b in BATCH_CHOICES] or BATCH_CHOICES
            epochs  = [e for e in ref.get("epochs", EPOCH_CHOICES) if e in EPOCH_CHOICES] or EPOCH_CHOICES

            # set refined space for Phase 2
            study.set_user_attr("refined", True)
            study.set_user_attr("lr_min", lr_min)
            study.set_user_attr("lr_max", lr_max)
            study.set_user_attr("batches", batches)
            study.set_user_attr("epochs",  epochs)

            cfg = data.get("config", {})
            cfg["lr"] = float(min(max(cfg.get("lr", (lr_min + lr_max) / 2), lr_min), lr_max))
            b_raw = int(cfg.get("batch", batches[0]))
            e_raw = int(cfg.get("epochs", epochs[0]))
            b_sel = min(batches, key=lambda x: abs(x - b_raw))
            e_sel = min(epochs,  key=lambda x: abs(x - e_raw))

            # enqueue phase-2 names (avoid Optuna dynamic categorical error)
            study.enqueue_trial({
                "lr_r2": cfg["lr"],
                "batch_r2": b_sel,
                "epochs_r2": e_sel,
            })

            print(f"Gemini refine [{j+1}/{LLM_REFINE_CALLS}] for {PATIENT}: "
                  f"space=({lr_min:.1e},{lr_max:.1e}), batches={batches}, epochs={epochs}; "
                  f"cfg={{'lr':{cfg['lr']}, 'batch':{b_sel}, 'epochs':{e_sel}}}")
        except Exception as e:
            print(f"⚠️ Gemini refine [{j+1}] skipped after retries: {e}")
            if study.user_attrs.get("refined", False):
                lr_min = float(study.user_attrs["lr_min"]); lr_max = float(study.user_attrs["lr_max"])
                batches = list(study.user_attrs["batches"]); epochs = list(study.user_attrs["epochs"])
                fallback = {"lr_r2": (lr_min + lr_max) / 2.0, "batch_r2": batches[0], "epochs_r2": epochs[0]}
                study.enqueue_trial(fallback)
                print(f"Enqueued fallback refined config: {fallback}")

    # ---- Phase 2 ----
    second_half = N_TRIALS - first_half
    if second_half > 0:
        print(f"Starting Optuna Phase 2 for {PATIENT} — trials={second_half}")
        study.optimize(objective, n_trials=second_half)

    # Save best
    best = study.best_trial
    best_cfg = {
        "lr":     best.params.get("lr_r2",     best.params.get("lr")),
        "batch":  best.params.get("batch_r2",  best.params.get("batch")),
        "epochs": best.params.get("epochs_r2", best.params.get("epochs")),
    }
    with open(os.path.join(out_dir, f"{PATIENT}_best.json"), "w") as f:
        json.dump({"best_value": best.value, "best_params": best_cfg}, f, indent=2)
    print(f"🏆 Best {PATIENT}: {best_cfg}  objective={best.value:.6f}")

    # Optional polish
    POLISH_EPOCHS = int(os.environ.get("POLISH_EPOCHS", "120"))
    if POLISH_EPOCHS > best_cfg["epochs"]:
        best_cfg_polish = dict(best_cfg); best_cfg_polish["epochs"] = POLISH_EPOCHS
        _ = fit_and_eval(best_cfg_polish, "polish_best")

print("\n🎉 All done! Aggregated CSV:", CSV_PATH)
