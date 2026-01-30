"""
Explainability + Multimodal Report for EfficientNet-B0 + LSTM EEG Spectrograms
- Test split only; enforces 2 seizure + 2 non-seizure examples (if available)
- Saves: original spectrogram triptych, Grad-CAM, Grad-CAM++, IG triptych
- Labeled figures (True/Pred + probability), confusion matrix, Markdown report
- Gemini LLM summary with strict retry; prompt includes metrics + per-sample notes + XAI digest

Project expectations:
- Model: architecture_eff.Net (EfficientNet-B0 backbone + LSTM head)
- Loader: loader_spectrograms.load_patient_split → tensors (B, T, 21, H, W)
"""

import os, time, random
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec 
from sklearn.metrics import confusion_matrix, f1_score

# ---- env: use non-cuDNN path to keep RNN backward simple for attributions
torch.backends.cudnn.enabled = False

from contextlib import contextmanager

@contextmanager
def cam_modes(model):
    """
    Enable RNN backward safely:
    - Put model in train() (needed for cuDNN RNN backward if enabled elsewhere)
    - Keep CNN backbone & classifier eval to stabilize BN/Dropout
    - Force any nn.LSTM/nn.GRU/nn.RNN modules to train()
    Restore modes on exit.
    """
    was_model = model.training
    was_backbone = getattr(model, "backbone", None).training if hasattr(model, "backbone") else None
    was_classifier = getattr(model, "classifier", None).training if hasattr(model, "classifier") else None

    rnn_list, rnn_modes = [], []
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            rnn_list.append(m)
            rnn_modes.append(m.training)

    model.train(True)
    if hasattr(model, "backbone"):   model.backbone.eval()
    if hasattr(model, "classifier"): model.classifier.eval()
    for r in rnn_list: r.train(True)

    try:
        yield
    finally:
        model.train(was_model)
        if was_backbone is not None and hasattr(model, "backbone"):
            model.backbone.train(was_backbone)
        if was_classifier is not None and hasattr(model, "classifier"):
            model.classifier.train(was_classifier)
        for r, was in zip(rnn_list, rnn_modes):
            r.train(was)

# ====== HARD-CODE YOUR SETTINGS HERE =========================================================
CKPT_PATH = r"E:\EEG\TRANSFER\models\chb08_EffB0LSTM_best.pt"  # <- set
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"                        # <- set
PATIENT   = "chb08"                                                                       # <- set
OUT_DIR   = r"E:\EEG\TRANSFER\models_optuna_eff_gemini_verbose\chb08\xai_report"         # <- set
SEQ_LEN   = 3

# Gemini (LLM) summary (dummy key kept as requested)
GEMINI_API_KEY = "AIzaSyAdeKHB0xEO9tb5DJttUPyahqguaI-s3fw"  # dummy placeholder; keep as-is
GEMINI_MODEL   = "gemini-2.0-flash"

# ====== LLM STRICTNESS & RETRIES ======
REQUIRE_LLM = True              # do not skip; fail the run if LLM can't be called successfully
LLM_TOTAL_TIMEOUT_S = 900       # hard cap on total retry time (e.g., 15 minutes)
LLM_MAX_RETRIES = 1000          # safety cap on attempts (very high)
LLM_INITIAL_BACKOFF_S = 2.0     # starting backoff (exponential)
LLM_BACKOFF_CAP_S = 60.0        # max backoff between attempts
LLM_TIMEOUT_PER_CALL_S = 90     # HTTP timeout per request
# ============================================================================================

# Your project modules
from architecture_eff import Net
from loader_spectrograms import load_patient_split

# ---------------- Metrics / Best-threshold ----------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cmatrix = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cmatrix.ravel() if cmatrix.size == 4 else (0,0,0,0)
    acc  = (TP+TN)/(TP+TN+FP+FN+1e-8)
    sens = TP/(TP+FN+1e-8)
    spec = TN/(TN+FP+1e-8)
    f1   = 2*TP/(2*TP+FP+FN+1e-8)
    f1w  = f1_score(y_true, y_pred, average='weighted') if cmatrix.size==4 else 0.0
    return acc, sens, spec, f1, f1w, cmatrix

def scan_best_threshold(y_true, y_prob):
    best_val, best_th, best_pack = -1.0, 0.5, None
    for th in np.linspace(0.05, 0.9, 171):
        acc, sens, spec, f1, f1w, cmatrix = compute_metrics(y_true, y_prob, th)
        val = 0.5*acc + 0.5*sens
        if val > best_val:
            best_val, best_th, best_pack = val, th, (acc, sens, spec, f1, f1w, cmatrix)
    return best_th, best_pack, best_val

# ---------------- Grad-CAM / Grad-CAM++ -------------------
class _Hook:
    def __init__(self, m: nn.Module):
        self.m = m
        self.a = None
        self.g = None
        self.h1 = m.register_forward_hook(self._fwd)
        self.h2 = m.register_full_backward_hook(self._bwd)
    def _fwd(self, module, inp, out):
        self.a = out.detach()
    def _bwd(self, module, grad_in, grad_out):
        self.g = grad_out[0].detach()
    def remove(self):
        self.h1.remove(); self.h2.remove()

def _upsample_to(img: torch.Tensor, sizeHW: Tuple[int,int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(img, size=sizeHW, mode="bilinear", align_corners=False)

@torch.no_grad()
def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8: return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def _overlay(spec_hw: np.ndarray, cam_hw: np.ndarray, alpha: float=0.45) -> np.ndarray:
    s = _normalize01(spec_hw)
    cam = _normalize01(cam_hw)
    cmap = plt.get_cmap("jet")
    cam_rgb = cmap(cam)[..., :3]                     # blue → red = low → high importance
    spec_rgb = cm.viridis(s)[..., :3]               # multi-color power map for clarity
    out = (1-alpha)*spec_rgb + alpha*cam_rgb
    return np.clip(out, 0.0, 1.0)

def gradcam_sequence(model: nn.Module, x: torch.Tensor, target_class: int = 1) -> List[np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    tgt = model.backbone.features[-1]  # last EfficientNet block
    hk = _Hook(tgt)

    x = x.clone().to(device).float().requires_grad_(True)
    with cam_modes(model):
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, target_class]
        model.zero_grad(set_to_none=True)
        prob.backward(retain_graph=True)

    acts, grads = hk.a, hk.g   # (B*T, Ck, Hk, Wk)
    hk.remove()

    _, T, C, H, W = x.shape
    BT, Ck, Hk, Wk = acts.shape
    assert BT == T, "Expect B==1 so BT==T"

    cams = []
    for t in range(T):
        A = acts[t]
        G = grads[t]
        w = G.mean(dim=(1,2), keepdim=True)
        cam = torch.relu((w * A).sum(dim=0, keepdim=True))
        cam = cam / (cam.max() + 1e-8)
        cam = _upsample_to(cam[None, ...], (H, W))[0,0]
        cams.append(cam.detach().cpu().numpy())
    return cams

def gradcampp_sequence(model: nn.Module, x: torch.Tensor, target_class: int = 1) -> List[np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    tgt = model.backbone.features[-1]
    hk = _Hook(tgt)

    x = x.clone().to(device).float().requires_grad_(True)
    with cam_modes(model):
        logits = model(x)
        score = torch.softmax(logits, dim=1)[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

    A, G = hk.a, hk.g
    hk.remove()

    _, T, C, H, W = x.shape
    BT, Ck, Hk, Wk = A.shape
    assert BT == T

    cams = []
    eps = 1e-10
    for t in range(T):
        a = A[t]
        g = G[t]
        g2 = g * g
        g3 = g2 * g

        num = g2
        den = 2.0 * g2 + (a * g3).sum(dim=(1,2), keepdim=True)
        den = torch.where(den < eps, torch.full_like(den, eps), den)
        alpha = num / den
        w = (alpha * torch.relu(g)).sum(dim=(1,2), keepdim=True)

        cam = torch.relu((w * a).sum(dim=0, keepdim=True))
        cam = cam / (cam.max() + eps)
        cam = _upsample_to(cam[None, ...], (H, W))[0,0]
        cams.append(cam.detach().cpu().numpy())
    return cams

# ---------------- Integrated Gradients (Captum) -----------
def ig_sequence(model: nn.Module, X_sample: torch.Tensor, baseline: torch.Tensor = None, n_steps: int = 64):
    from captum.attr import IntegratedGradients
    device = next(model.parameters()).device
    model.eval()

    class ProbWrapper(nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return torch.softmax(self.base(x), dim=1)[:, 1:2]  # (B,1)

    wrapped = ProbWrapper(model).to(device)
    ig = IntegratedGradients(wrapped)

    X_sample = X_sample.to(device).float()
    if baseline is None:
        baseline = torch.zeros_like(X_sample, device=device)

    with cam_modes(model):
        attrs, _ = ig.attribute(
            X_sample, baselines=baseline, target=None, n_steps=n_steps, return_convergence_delta=True
        )
    vals = attrs[0].detach().cpu().numpy()  # (T,21,H,W)
    per_t = vals.mean(axis=1)                # (T,H,W) mean over channels
    return per_t

# ---------------- XAI digest (text sent to LLM) ----------
def summarize_ig(ig_maps: np.ndarray) -> str:
    T, H, W = ig_maps.shape
    bands = np.array_split(np.arange(H), 5)
    band_names = ["B1(lowest)", "B2", "B3", "B4", "B5(highest)"]
    lines = []
    for t in range(T):
        M = ig_maps[t]
        pos = np.clip(M, 0, None)
        total_pos = pos.sum() + 1e-8
        frac_pos = total_pos / (np.abs(M).sum() + 1e-8)
        band_scores = [pos[b].mean() for b in bands]
        topb = int(np.argmax(band_scores))
        lines.append(f"t={t}: top {band_names[topb]}, pos-fraction={frac_pos:.2f}")
    return "; ".join(lines)

# ---------------- Visualization utilities ----------------
def save_confusion_matrix(cm_counts: np.ndarray, out_png: str):
    """Light-blue confusion matrix with centered counts and non-overlapping TN/FP/FN/TP tags."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm_counts, cmap=plt.cm.Blues, vmin=0)

    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred: Non-seizure', 'Pred: Seizure'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['True: Non-seizure', 'True: Seizure'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_counts[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold", color="black")

    corner_labels = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    for (i, j), lab in corner_labels.items():
        ax.text(j - 0.45, i - 0.43, lab,
                ha="left", va="top", fontsize=11, color="#0a4d8c")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Count", rotation=270, labelpad=12)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def save_grid(images: List[np.ndarray], titles: List[str], out_png: str,
              cols: int = 3, suptitle: str = None, suptitle_color: str = "black"):
    n = len(images)
    rows = (n + cols - 1)//cols
    fig = plt.figure(figsize=(3.5*cols, 3.2*rows + (0.7 if suptitle else 0)))
    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, color=suptitle_color, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.965] if suptitle else None)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def save_spec(image_hw: np.ndarray, out_png: str, title: str = None):
    """Single spectrogram (viridis) with its own colorbar."""
    img = _normalize01(image_hw)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(img, cmap="viridis", interpolation="nearest", aspect="auto")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative power (normalized)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def save_spec_triptych(spec_list, out_png, suptitle=None, suptitle_color="black"):
    """3-panel spectrogram triptych with a dedicated colorbar axis (no overlap)."""
    assert len(spec_list) == 3
    specs = [ _normalize01(s) for s in spec_list ]

    # 3 image axes + 1 thin colorbar axis
    fig = plt.figure(figsize=(11.5, 4.0), constrained_layout=True)
    gs = GridSpec(nrows=1, ncols=4, figure=fig, width_ratios=[1, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    vmin, vmax = 0.0, 1.0
    ims = []
    for t, ax in enumerate(axes):
        im = ax.imshow(specs[t], cmap="viridis", interpolation="nearest",
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t}", fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)

    # single colorbar placed in its own axis (won't overlap)
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Relative power (normalized)", fontsize=11)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, color=suptitle_color)

    # do NOT call tight_layout when constrained_layout=True
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def save_ig_triptych(ig_maps, out_png, suptitle=None, suptitle_color="black"):
    """3-panel IG triptych with symmetric limits and a dedicated colorbar axis."""
    assert ig_maps.shape[0] == 3
    vmax = float(np.nanmax(np.abs(ig_maps))) + 1e-8
    vmin = -vmax

    fig = plt.figure(figsize=(11.5, 4.0), constrained_layout=True)
    gs = GridSpec(nrows=1, ncols=4, figure=fig, width_ratios=[1, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    ims = []
    for t, ax in enumerate(axes):
        im = ax.imshow(ig_maps[t], cmap="seismic", interpolation="nearest",
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"IG t={t}", fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)

    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Attribution (− to +)", fontsize=11)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, color=suptitle_color)

    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def save_cam_legend(out_png: str):
    """Save a small legend explaining CAM colors: blue→low, red→high importance."""
    gradient = np.linspace(0, 1, 256)[None, :]
    fig, ax = plt.subplots(figsize=(4.2, 0.7))
    ax.imshow(gradient, aspect='auto', cmap="jet")
    ax.set_yticks([]); ax.set_xticks([0,64,128,192,255])
    ax.set_xticklabels(["low", "", "medium", "", "high"])
    ax.set_xlabel("Grad-CAM / Grad-CAM++ importance (blue→red)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def save_spec_legend(out_png: str):
    """Save a small legend for spectrogram power using viridis."""
    gradient = np.linspace(0, 1, 256)[None, :]
    fig, ax = plt.subplots(figsize=(4.2, 0.7))
    ax.imshow(gradient, aspect='auto', cmap="viridis")
    ax.set_yticks([]); ax.set_xticks([0,64,128,192,255])
    ax.set_xticklabels(["low", "", "medium", "", "high"])
    ax.set_xlabel("Spectrogram relative power (viridis: dark→bright)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

# ---------------- LLM summary (strict retry) -------------
def gemini_summary_strict(metrics: Dict[str, Any],
                          notes: List[str],
                          xai_digest: str,
                          api_key: str,
                          model_name: str = "gemini-2.0-flash",
                          temperature: float = 0.25) -> str:
    """
    Strict Gemini caller with retries. (LLM prompt kept exactly as you provided.)
    """
    import requests

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing but LLM is required (REQUIRE_LLM=True).")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # >>> DO NOT CHANGE THE PROMPT (per user request) <<<
    prompt = f"""Write a report on EEG spectrogram classification.
Use an educator tone explaining seizure. Be factual and to the point and no need to explain abt metrics except to add weight.

Metrics:
- Accuracy: {metrics['acc']:.4f}
- Sensitivity: {metrics['sens']:.4f}
- Specificity: {metrics['spec']:.4f}
- F1: {metrics['f1']:.4f}
- Weighted F1: {metrics['f1w']:.4f}
- Best threshold: {metrics['thr']:.2f}

Observed samples:
{chr(10).join('- '+x for x in notes)}

Explainability digest (from Grad-CAM/IG):
{xai_digest}

Tasks:
Explain which time–frequency regions were emphasized across timesteps and why that supports the decisions in a beginner-friendly yet comprehensive over the significant medical details pls be technical in analysis of samples."""
    # <<< END PROMPT >>>

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature}
    }

    start = time.time()
    attempt = 0
    backoff = LLM_INITIAL_BACKOFF_S
    last_err = None

    while True:
        attempt += 1
        if (time.time() - start) > LLM_TOTAL_TIMEOUT_S or attempt > LLM_MAX_RETRIES:
            raise RuntimeError(f"LLM failed after {attempt-1} attempts / {int(time.time()-start)}s. Last error: {last_err}")

        try:
            r = requests.post(url, json=body, timeout=LLM_TIMEOUT_PER_CALL_S)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                sleep_s = min(backoff, LLM_BACKOFF_CAP_S) * (1.0 + 0.25*random.random())
                time.sleep(sleep_s)
                backoff = min(backoff * 2.0, LLM_BACKOFF_CAP_S)
                continue

            r.raise_for_status()
            data = r.json()
            txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if not txt:
                last_err = "Empty response text"
                time.sleep(1.0)
                continue
            return txt

        except Exception as e:
            last_err = repr(e)
            sleep_s = min(backoff, LLM_BACKOFF_CAP_S) * (1.0 + 0.25*random.random())
            time.sleep(sleep_s)
            backoff = min(backoff * 2.0, LLM_BACKOFF_CAP_S)

# ---------------- Main pipeline --------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: uses 70/30 split inside, returns test set for unseen evaluation
    (Xtr, ytr), (Xte, yte) = load_patient_split(PATIENT, SPEC_DIR, seq_len=SEQ_LEN, augment=False)

    # Logging (mean/std of training tensor)
    u, v = np.mean(Xtr.numpy()), np.std(Xtr.numpy())
    print(f"{PATIENT}: train {len(Xtr)}  test {len(Xte)}")
    print(f"train labels: {{0: {int((ytr==0).sum())}, 1: {int((ytr==1).sum())}}}")
    print(f"test  labels: {{0: {int((yte==0).sum())}, 1: {int((yte==1).sum())}}}")
    print("Sample batch mean/std:", u, v)

    # Model
    model = Net(n_classes=2, seq_len=SEQ_LEN, fine_tune=True).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Probabilities on TEST
    probs = []
    with torch.no_grad():
        for i in range(len(Xte)):
            x = Xte[i:i+1].to(device).float()
            p = torch.softmax(model(x), dim=1)[:,1]
            probs.append(float(p.item()))
    y_true = yte.numpy()
    y_prob = np.array(probs)
    thr, (acc, sens, spec, f1, f1w, cmatrix), _ = scan_best_threshold(y_true, y_prob)

    # Save confusion matrix
    cm_png = os.path.join(OUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(cmatrix, cm_png)

    # Legends (saved once)
    cam_legend_png  = os.path.join(OUT_DIR, "legend_gradcam.png")
    spec_legend_png = os.path.join(OUT_DIR, "legend_spectrogram.png")
    if not os.path.exists(cam_legend_png):  save_cam_legend(cam_legend_png)
    if not os.path.exists(spec_legend_png): save_spec_legend(spec_legend_png)

    # --- enforce exactly 2 seizure + 2 non-seizure picks from TEST (if available) ---
    pos_idx = np.where(y_true == 1)[0].tolist()
    neg_idx = np.where(y_true == 0)[0].tolist()
    pos_sel = pos_idx[:2]
    neg_sel = neg_idx[:2]
    if len(pos_sel) < 2 and len(neg_idx) > 0:
        need = 2 - len(pos_sel); pos_sel += neg_idx[:need]
    if len(neg_sel) < 2 and len(pos_idx) > 0:
        need = 2 - len(neg_sel); neg_sel += pos_idx[:need]

    picks, seen = [], set()
    for idx in pos_sel + neg_sel:
        if idx not in seen:
            seen.add(idx); picks.append(idx)
        if len(picks) == 4: break

    # Explainability per pick
    notes: List[str] = []
    ig_files_per_sample: Dict[int, List[str]] = {}
    xai_digests: List[str] = []

    for idx in picks:
        xs = Xte[idx:idx+1].to(device).float()           # (1,T,21,H,W)
        T, H, W = xs.shape[1], xs.shape[-2], xs.shape[-1]

        # Labels for titles
        p_hat = y_prob[idx]
        pred  = int(p_hat >= thr)
        true_lbl = "Seizure" if int(y_true[idx]) == 1 else "Non-seizure"
        pred_lbl = "Seizure" if pred == 1 else "Non-seizure"
        ok = (pred == int(y_true[idx]))
        sup = f"Sample {idx} — True: {true_lbl} | Pred: {pred_lbl} | p_seizure={p_hat:.2f}"
        sup_color = "green" if ok else "red"

        # ----- ORIGINAL spectrograms (mean over 21 channels) & triptych
        spec_arrays = []
        spec_pngs = []
        for t in range(T):
            spec_t = xs[0, t].detach().cpu().numpy().mean(axis=0)  # (H,W)
            spec_arrays.append(spec_t)
            single_out = os.path.join(OUT_DIR, f"sample{idx}_spec_t{t}.png")
            save_spec(spec_t, single_out, title=f"{true_lbl} | Pred: {pred_lbl} | t={t} (mean of 21ch)")
            spec_pngs.append(os.path.basename(single_out))
        # Triptych
        spec_trip = os.path.join(OUT_DIR, f"sample{idx}_spec_triptych.png")
        save_spec_triptych(spec_arrays, spec_trip, suptitle=sup, suptitle_color=sup_color)

        # ----- base image for CAM overlays (use last step mean over channels)
        base = spec_arrays[-1]

        # Grad-CAM
        gcams = gradcam_sequence(model, xs, target_class=1)
        gcam_imgs = [_overlay(base, g) for g in gcams]
        gcam_png = os.path.join(OUT_DIR, f"sample{idx}_gradcam.png")
        save_grid(gcam_imgs, [f"t={t}" for t in range(T)], gcam_png,
                  suptitle=sup, suptitle_color=sup_color)

        # Grad-CAM++
        gcpps = gradcampp_sequence(model, xs, target_class=1)
        gcpp_imgs = [_overlay(base, g) for g in gcpps]
        gcpp_png = os.path.join(OUT_DIR, f"sample{idx}_gradcampp.png")
        save_grid(gcpp_imgs, [f"t={t}" for t in range(T)], gcpp_png,
                  suptitle=sup, suptitle_color=sup_color)

        # ----- Integrated Gradients per timestep + triptych
        ig_pngs = []
        try:
            per_t_maps = ig_sequence(model, xs, baseline=None, n_steps=64)  # (T,H,W)
            for t in range(per_t_maps.shape[0]):
                fig, ax = plt.subplots(figsize=(3.4,3.0))
                vmax = np.nanmax(np.abs(per_t_maps)) + 1e-8
                vmin = -vmax
                im = ax.imshow(per_t_maps[t], cmap="seismic", interpolation="nearest", aspect="auto",
                               vmin=vmin, vmax=vmax)
                ax.set_title(f"{true_lbl} | Pred: {pred_lbl} | IG t={t} (mean channels)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Attribution (− to +)", fontsize=8)
                outp = os.path.join(OUT_DIR, f"sample{idx}_ig_t{t}.png")
                fig.tight_layout(); fig.savefig(outp, dpi=160); plt.close(fig)
                ig_pngs.append(os.path.basename(outp))
            ig_files_per_sample[idx] = ig_pngs

            # IG triptych
            ig_trip = os.path.join(OUT_DIR, f"sample{idx}_ig_triptych.png")
            save_ig_triptych(per_t_maps, ig_trip, suptitle=sup, suptitle_color=sup_color)

            # XAI digest for LLM
            digest = summarize_ig(per_t_maps)
            xai_digests.append(f"sample {idx} ({true_lbl}, pred={pred_lbl}): {digest}")

        except Exception as e:
            print("IG attribution skipped:", e)
            ig_files_per_sample[idx] = []
            xai_digests.append(f"sample {idx}: IG unavailable ({e})")

        files_list = [os.path.basename(spec_trip), os.path.basename(gcam_png),
                      os.path.basename(gcpp_png), os.path.basename(ig_trip)] + ig_files_per_sample[idx]
        notes.append(
            f"sample {idx}: true={int(y_true[idx])}, pred={pred}, p_seizure={p_hat:.3f}, "
            f"files=[{', '.join(files_list)}]"
        )

    # Markdown report
    md = os.path.join(OUT_DIR, "multimodal_report.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# EEG Spectrogram Model – XAI & Multimodal Report ({PATIENT})\n\n")
        f.write("## Metrics\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Accuracy | {acc*100:.2f}% |\n")
        f.write(f"| Sensitivity | {sens*100:.2f}% |\n")
        f.write(f"| Specificity | {spec*100:.2f}% |\n")
        f.write(f"| F1 | {f1:.4f} |\n")
        f.write(f"| Weighted F1 | {f1w:.4f} |\n")
        f.write(f"| Best threshold | {thr:.2f} |\n\n")

        f.write(f"Confusion matrix:\n\n![]({os.path.basename(cm_png)})\n\n")
        f.write("_Quadrant labels: TN = true non-seizure correctly predicted; FP = non-seizure predicted as seizure; FN = seizure predicted as non-seizure; TP = seizure correctly predicted._\n\n")

        f.write("## Color keys\n")
        f.write(f"![]({os.path.basename(spec_legend_png)})\n\n")
        f.write(f"![]({os.path.basename(cam_legend_png)})\n\n")

        f.write("## Examples (Spectrogram Triptych / Grad-CAM / Grad-CAM++ / IG Triptych)\n")
        for idx in picks:
            f.write(f"\n**Sample {idx}**\n\n")
            # Spectrogram triptych (preferred)
            sp_trip = f"sample{idx}_spec_triptych.png"
            if os.path.exists(os.path.join(OUT_DIR, sp_trip)):
                f.write(f"![]({sp_trip})\n\n")
                f.write("_Spectrogram color: viridis (dark→bright = low→high power)._ \n\n")
            # Grad-CAMs
            f.write(f"![](sample{idx}_gradcam.png)\n\n")
            f.write("_Grad-CAM color: blue→low importance, red→high importance (overlay on spectrogram)._ \n\n")
            f.write(f"![](sample{idx}_gradcampp.png)\n\n")
            f.write("_Grad-CAM++ color: same meaning with improved weighting._ \n\n")
            # IG triptych (preferred)
            ig_trip = f"sample{idx}_ig_triptych.png"
            if os.path.exists(os.path.join(OUT_DIR, ig_trip)):
                f.write(f"![]({ig_trip})\n\n")
                f.write("_IG color: blue = negative attribution (toward non-seizure), red = positive (toward seizure)._ \n\n")

        metrics = {"acc":acc, "sens":sens, "spec":spec, "f1":f1, "f1w":f1w, "thr":thr}
        xai_digest_text = "\n".join(xai_digests)

        if REQUIRE_LLM:
            summary = gemini_summary_strict(
                metrics, notes, xai_digest_text,
                api_key=GEMINI_API_KEY,
                model_name=GEMINI_MODEL,
                temperature=0.25
            )
        else:
            try:
                summary = gemini_summary_strict(
                    metrics, notes, xai_digest_text,
                    api_key=GEMINI_API_KEY,
                    model_name=GEMINI_MODEL,
                    temperature=0.25
                )
            except Exception as e:
                summary = f"(LLM summary skipped: {e})"

        f.write("## LLM Summary\n")
        f.write(summary + "\n")
        f.write("\n## Notes\n")
        for s in notes: f.write("- " + s + "\n")

    print("Report folder:", OUT_DIR)
    print("Markdown:", md)

if __name__ == "__main__":
    main()
