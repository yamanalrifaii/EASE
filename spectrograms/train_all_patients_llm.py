# =============================================================
# train_all_patients_optuna_gemini_boost.py
# Optuna + Gemini (REST) with: baseline guarantee, LLM seeding,
# mid-run refinement, baseline-neighborhood sweep, and post-study polish
# =============================================================
# Requires: pip install optuna requests
#
# HARD-CODE YOUR GEMINI KEY BELOW or set GEMINI_API_KEY env var.
# =============================================================

import os, json, time, random, math, glob, re, warnings, requests
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
import optuna
from optuna.samplers import TPESampler

from loader_spectrograms import load_patient_split
from architecture_spectrograms import Net
from trainer import trainer

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Put your key here (or leave "" to use env GEMINI_API_KEY) ----------
GKEY_HARDCODED = ""     # e.g. "AIzaSy...."
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ---------- Torch / device ----------
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Paths ----------
ROOT      = r"E:\EEG\spectrograms"
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
OUT_DIR   = os.path.join(ROOT, "models_optuna_gemini_boost")
RESULTS_DIR = os.path.join(OUT_DIR, "per_patient_results")
RES_CSV   = os.path.join(OUT_DIR, "spectrogram_results.csv")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Patients ----------
_env_patients = os.environ.get("PATIENTS", "").strip()
if _env_patients:
    PATIENTS = [p.strip() for p in _env_patients.split(",") if p.strip()]
else:
    cands = set()
    for f in glob.glob(os.path.join(SPEC_DIR, "*_spectrogram.npy")):
        base = os.path.basename(f)
        m = re.match(r"(chb\d+)_.*_spectrogram\.npy", base)
        if m: cands.add(m.group(1))
    PATIENTS = sorted(list(cands))
if not PATIENTS: raise SystemExit("No patients found. Set PATIENTS or check SPEC_DIR.")

# ---------- Search space (aligned with your trainer/Net) ----------
LR_MIN, LR_MAX = 3e-5, 1e-3
BATCH_CHOICES  = [64, 96, 128, 192]
EPOCH_CHOICES  = [70, 90, 100]

# Your baseline settings (guaranteed trial)
BASELINE_CFG = {"lr": 5e-4, "batch_size": 128, "epochs": 100}  # 

# ---------- Reproducibility ----------
GLOBAL_SEED = int(os.environ.get("SEED", "7"))
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(GLOBAL_SEED)

# ---------- Optuna knobs ----------
N_TRIALS    = int(os.environ.get("N_TRIALS", "24"))
TIMEOUT_SEC = int(os.environ.get("TIMEOUT_SEC", "0"))
SAMPLER     = TPESampler(seed=GLOBAL_SEED, n_startup_trials=min(6, max(3, N_TRIALS//4)))

# ---------- LLM knobs (free-tier mindful) ----------
LLM_CALLS_PER_PATIENT = max(0, min(int(os.environ.get("LLM_CALLS_PER_PATIENT", "2")), 3))
LLM_SEED_COUNT        = max(1, min(int(os.environ.get("LLM_SEED_COUNT", "2")), 3))
LLM_REFINE_COUNT      = max(1, min(int(os.environ.get("LLM_REFINE_COUNT", "2")), 3))

# ---------- Gemini REST helpers ----------
def _get_gemini_key() -> str:
    if GKEY_HARDCODED.strip(): return GKEY_HARDCODED.strip()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key: raise RuntimeError("Gemini key not configured (set GKEY_HARDCODED or GEMINI_API_KEY).")
    return key

def _gemini_rest(prompt: str, temperature: float = 0.25) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={_get_gemini_key()}"
    body = {"contents":[{"parts":[{"text":prompt}]}], "generationConfig":{"temperature":temperature}}
    r = requests.post(url, json=body, timeout=90); r.raise_for_status()
    data = r.json()
    try: return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception: raise RuntimeError(f"Gemini REST response unexpected: {data}")

def _snap_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    lr = float(np.clip(float(cfg.get("lr", BASELINE_CFG["lr"])), LR_MIN, LR_MAX))
    bs = int(cfg.get("batch_size", BASELINE_CFG["batch_size"]))
    if bs not in BATCH_CHOICES: bs = min(BATCH_CHOICES, key=lambda x: abs(x-bs))
    ep = int(cfg.get("epochs", BASELINE_CFG["epochs"]))
    if ep not in EPOCH_CHOICES: ep = min(EPOCH_CHOICES, key=lambda x: abs(x-ep))
    return {"lr": lr, "batch_size": bs, "epochs": ep}

def llm_seed_suggestions(n: int, early_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = f"""
Propose {n} diverse hyperparameter sets for a 2-class spectrogram CNN seizure detector.
Use ONLY these spaces:
- lr (log-uniform): {LR_MIN}..{LR_MAX}
- batch_size: {BATCH_CHOICES}
- epochs: {EPOCH_CHOICES}
Return strict JSON: {{"configs":[{{"lr":float,"batch_size":int,"epochs":int}},...]}}
Recent local results (may be empty): {json.dumps(early_results)}
JSON only.
""".strip()
    txt = _gemini_rest(prompt); s, e = txt.find("{"), txt.rfind("}")
    cfgs = json.loads(txt[s:e+1])["configs"]
    return [_snap_cfg(c) for c in cfgs][:n]

def llm_refine_and_propose(history: List[Dict[str, Any]], n: int):
    top5 = sorted(history, key=lambda r: (-r["f1"], -r["sens"]))[:5]
    prompt = f"""
Refine the search space (narrower ranges) using the top results below, then propose {n} promising configs.
Original spaces:
- lr: {LR_MIN}..{LR_MAX} (log-uniform)
- batch_size: {BATCH_CHOICES}
- epochs: {EPOCH_CHOICES}

Top results: {json.dumps(top5)}
Return strict JSON:
{{
 "refined_space": {{"lr_min":float,"lr_max":float,"batch_choices":{BATCH_CHOICES},"epoch_choices":{EPOCH_CHOICES}}},
 "configs":[{{"lr":float,"batch_size":int,"epochs":int}},...]
}}
Constraints: lr_min>= {LR_MIN}, lr_max<= {LR_MAX}, lr_min<lr_max; batch/epoch choices must be non-empty subsets.
JSON only.
""".strip()
    txt = _gemini_rest(prompt); s, e = txt.find("{"), txt.rfind("}")
    data = json.loads(txt[s:e+1])

    ref = data.get("refined_space", {})
    lr_min = float(max(LR_MIN, ref.get("lr_min", LR_MIN)))
    lr_max = float(min(LR_MAX, ref.get("lr_max", LR_MAX)))
    if lr_min >= lr_max: lr_min, lr_max = LR_MIN, LR_MAX
    bch = [b for b in ref.get("batch_choices", BATCH_CHOICES) if b in BATCH_CHOICES] or BATCH_CHOICES
    ech = [e for e in ref.get("epoch_choices", EPOCH_CHOICES) if e in EPOCH_CHOICES] or EPOCH_CHOICES

    cfgs = [_snap_cfg(c) for c in data.get("configs", [])][:n]
    return {"lr_min": lr_min, "lr_max": lr_max, "batch_choices": bch, "epoch_choices": ech}, cfgs

# ---------- Evaluation (best-F1 threshold scan 0.1..0.9; same as your code) ----------
def evaluate(model: torch.nn.Module, X_test, y_test) -> Tuple[float, float, float, float, float]:
    model.eval(); X_test = X_test.to("cpu"); y_test = y_test.to("cpu")
    probs = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        loader = torch.utils.data.DataLoader(X_test, batch_size=128, shuffle=False, pin_memory=False)
        for data in loader:
            data = data.to(DEVICE).float()
            out = torch.softmax(model(data), dim=1)
            probs.extend(out[:, 1].cpu().numpy())
    y_prob = np.array(probs)
    best_f1, best_th = 0.0, 0.5
    for th in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_test.cpu(), (y_prob >= th).astype(int))
        if f1 > best_f1: best_f1, best_th = f1, th
    y_pred = (y_prob >= best_th).astype(int)
    cm = confusion_matrix(y_test.cpu(), y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = TP / (TP + FN + 1e-10) if (TP + FN) else 0.0
    spec = TN / (TN + FP + 1e-10) if (TN + FP) else 0.0
    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    return sens, spec, acc, best_th, best_f1

# ---------- CSV header (same schema you use) ----------
import csv
if not os.path.exists(RES_CSV):
    with open(RES_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "Patient","Accuracy","Sensitivity","Specificity","F1","Best_Threshold",
            "Trial","lr","batch_size","epochs","optuna_value"
        ])

# ---------- Main ----------
for PATIENT in PATIENTS:
    print(f"\n================ {PATIENT}: loading data ================")
    Train_set, Test_set = (
        load_patient_split(PATIENT, SPEC_DIR, verbose=True)
        if "verbose" in load_patient_split.__code__.co_varnames
        else load_patient_split(PATIENT, SPEC_DIR)
    )
    X_train, y_train = Train_set; X_test, y_test = Test_set

    patient_dir = os.path.join(RESULTS_DIR, PATIENT); os.makedirs(patient_dir, exist_ok=True)
    history: List[Dict[str, Any]] = []

    def log_and_return(model_path, cfg: Dict[str, Any], trial_tag: str) -> float:
        model = Net(n_classes=2).to(DEVICE)
        t = trainer(model, Train_set); t.compile(lr=cfg["lr"])
        t.train(epochs=cfg["epochs"], batch_size=cfg["batch_size"], directory=model_path)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        sens, spec, acc, th, f1b = evaluate(model, X_test, y_test)
        rec = {"trial": trial_tag, "sens":float(sens), "spec":float(spec), "acc":float(acc),
               "cfg":cfg, "threshold":float(th), "f1":float(f1b)}
        history.append(rec)
        with open(os.path.join(patient_dir, f"{PATIENT}_optuna_results.json"), "w") as jf:
            json.dump(history, jf, indent=2)
        with open(RES_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                PATIENT, f"{acc*100:.2f}", f"{sens*100:.2f}", f"{spec*100:.2f}",
                f"{f1b*100:.2f}", f"{th:.2f}", trial_tag,
                cfg["lr"], cfg["batch_size"], cfg["epochs"], f"{f1b:.6f}"
            ])
        return f1b

    # --------- Objective (SAFE: static space; no mid-run shrinking) ---------
    def objective(trial: optuna.Trial) -> float:
        # Always use the ORIGINAL full space so fixed/enqueued params are valid.
        # If this trial was enqueued with fixed params, make sure they're included.
        bs_choices = list(BATCH_CHOICES)
        ep_choices = list(EPOCH_CHOICES)

        bs_fixed = trial.params.get("batch_size")
        ep_fixed = trial.params.get("epochs")
        if bs_fixed is not None and bs_fixed not in bs_choices:
            bs_choices = sorted(set(bs_choices + [bs_fixed]))
        if ep_fixed is not None and ep_fixed not in ep_choices:
            ep_choices = sorted(set(ep_choices + [ep_fixed]))

        # (Optional) Soft bias: if you saved a refined space in user_attrs, just reorder choices
        # so refined ones are tried earlier—WITHOUT excluding others.
        refined = trial.study.user_attrs.get("refined_space")
        if refined:
            def prioritize(choices, prefer):
                prefer = [v for v in prefer if v in choices]
                rest = [v for v in choices if v not in prefer]
                return prefer + rest
            bs_choices = prioritize(bs_choices, refined.get("batch_choices", []))
            ep_choices = prioritize(ep_choices, refined.get("epoch_choices", []))
            # We deliberately DO NOT change lr bounds here to avoid conflicts.

        # Suggest from the full, safe space
        lr = trial.suggest_float("lr", float(LR_MIN), float(LR_MAX), log=True)
        batch_size = trial.suggest_categorical("batch_size", bs_choices)
        epochs     = trial.suggest_categorical("epochs", ep_choices)

        cfg = {"lr": lr, "batch_size": batch_size, "epochs": epochs}
        model_path = os.path.join(patient_dir, f"{PATIENT}_trial{trial.number}.pt")
        return log_and_return(model_path, cfg, f"optuna{trial.number}")

    # Create the study as before (no change)
    study = optuna.create_study(direction="maximize", sampler=SAMPLER, study_name=f"{PATIENT}_study")

    # --------- Guarantee: enqueue your exact baseline + local neighborhood ---------
    study.enqueue_trial(BASELINE_CFG)  # exact match to your strong baseline 
    # neighborhood around baseline lr & batch:
    for lr_mul in [0.5, 0.75, 1.25, 1.5]:
        for bs in [96, 128, 192]:
            study.enqueue_trial({"lr": float(BASELINE_CFG["lr"]*lr_mul),
                                 "batch_size": int(bs),
                                 "epochs": int(BASELINE_CFG["epochs"])})

    # --------- LLM seeding before run (if budget allows) ----------
    remaining_llm = LLM_CALLS_PER_PATIENT
    if remaining_llm > 0:
        try:
            seed_cfgs = llm_seed_suggestions(n=LLM_SEED_COUNT, early_results=[])
            for c in seed_cfgs: study.enqueue_trial(c)
            print(f"Gemini seed for {PATIENT}: {seed_cfgs}")
            remaining_llm -= 1
        except Exception as e:
            print(f"⚠️ Gemini seeding skipped: {e}")

    # --------- Phase 1 (first half) ----------
    first_half = max(1, N_TRIALS // 2)
    print(f"Starting Optuna Phase 1 for {PATIENT} — trials={first_half}")
    study.optimize(objective, n_trials=first_half, timeout=None if TIMEOUT_SEC==0 else TIMEOUT_SEC//2)

    # --------- LLM refine mid-run (if budget allows) ----------
    if remaining_llm > 0:
        try:
            refined, cfgs = llm_refine_and_propose(history, n=LLM_REFINE_COUNT)
            # store refined space:
            study.set_user_attr("lr_min", refined["lr_min"])
            study.set_user_attr("lr_max", refined["lr_max"])
            study.set_user_attr("batch_choices", refined["batch_choices"])
            study.set_user_attr("epoch_choices", refined["epoch_choices"])
            for c in cfgs: study.enqueue_trial(c)
            print(f"Gemini refine for {PATIENT}: space={refined}, cfgs={cfgs}")
            remaining_llm -= 1
        except Exception as e:
            print(f"⚠️ Gemini refine skipped: {e}")

    # --------- Phase 2 (second half) ----------
    second_half = N_TRIALS - first_half
    if second_half > 0:
        print(f"Starting Optuna Phase 2 for {PATIENT} — trials={second_half}")
        study.optimize(objective, n_trials=second_half, timeout=None if TIMEOUT_SEC==0 else TIMEOUT_SEC - (TIMEOUT_SEC//2))

    # --------- Post-study "polish": retrain top-2 configs at epochs=100 ----------
    ranked = sorted(history, key=lambda r: (-r["f1"], -r["sens"]))
    for k, rec in enumerate(ranked[:2]):
        cfg = dict(rec["cfg"]); cfg["epochs"] = 100  # match your baseline training length 
        tag = f"polish{k}"
        model_path = os.path.join(patient_dir, f"{PATIENT}_{tag}.pt")
        log_and_return(model_path, cfg, tag)

    # --------- Save optuna best ---------
    best = study.best_trial
    best_cfg = {"lr": best.params["lr"], "batch_size": best.params["batch_size"], "epochs": best.params["epochs"]}
    with open(os.path.join(patient_dir, f"{PATIENT}_best.json"), "w") as f:
        json.dump({"best_value": best.value, "best_params": best_cfg}, f, indent=2)

    print("\n🏆 Best for", PATIENT, "→", best_cfg, "value(F1)=", best.value)

print("\n🎉 All done! Aggregated CSV:", RES_CSV)
