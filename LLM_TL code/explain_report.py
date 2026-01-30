"""
Explainability + Multimodal Report for EfficientNet-B0 + LSTM EEG Spectrograms
- Grad-CAM, Grad-CAM++, optional SHAP (if installed)
- Uses TEST split only; enforces exactly 2 seizure + 2 non-seizure examples (if available)
- Metrics, best-threshold scan, confusion matrix, Markdown report
- Optional Gemini LLM textual summary

Compatibility with your repo:
- Model: architecture_eff.Net (EfficientNet-B0 backbone + LSTM head)
- Loader: loader_spectrograms.load_patient_split  → tensors (B, T, 21, H, W)
"""

import os, json, math, time
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

import torch
torch.backends.cudnn.enabled = False


# put near the top of the file, after imports
from contextlib import contextmanager

@contextmanager
def cam_modes(model):
    """
    Enable cuDNN RNN backward safely:
    - keep backbone & classifier in eval()
    - put temporal (LSTM) in train() to allow backward
    and restore original modes afterward.
    """
    # record current modes
    was_model = model.training
    was_backbone = getattr(model.backbone, 'training', None)
    was_temporal = getattr(model, 'temporal', None) and model.temporal.training
    was_classifier = getattr(model, 'classifier', None) and model.classifier.training

    # set safe modes
    model.train(True)                # allow cuDNN RNN backward
    model.backbone.eval()            # freeze BN/Dropout in CNN
    if hasattr(model, 'classifier'): 
        model.classifier.eval()      # disable Dropout in head
    if hasattr(model, 'temporal'):   
        model.temporal.train(True)   # cuDNN LSTM needs train()

    try:
        yield
    finally:
        # restore original modes
        model.train(was_model)
        if was_backbone is not None: model.backbone.train(was_backbone)
        if hasattr(model, 'classifier') and was_classifier is not None:
            model.classifier.train(was_classifier)
        if hasattr(model, 'temporal') and was_temporal is not None:
            model.temporal.train(was_temporal)


# Optional SHAP
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# ====== HARD-CODE YOUR SETTINGS HERE =========================================================
CKPT_PATH = r"E:\EEG\TRANSFER\models_optuna_eff_gemini_verbose\chb20\chb20_optuna3.pt"  # <- set
SPEC_DIR  = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"                        # <- set
PATIENT   = "chb20"                                                                       # <- set
OUT_DIR   = r"E:\EEG\TRANSFER\models_optuna_eff_gemini_verbose\chb20\xai_report"         # <- set
SEQ_LEN   = 3

# Gemini (LLM) summary; leave empty to skip
GEMINI_API_KEY = "AIzaSyAdeKHB0xEO9tb5DJttUPyahqguaI-s3fw"  # or leave "" to skip
GEMINI_MODEL   = "gemini-2.0-flash"
# ============================================================================================

# Your project modules
from architecture_eff import Net
from loader_spectrograms import load_patient_split

# ---------------- Metrics / Best-threshold (matches your training objective) ----------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc  = (TP+TN)/(TP+TN+FP+FN+1e-8)
    sens = TP/(TP+FN+1e-8)
    spec = TN/(TN+FP+1e-8)
    f1   = 2*TP/(2*TP+FP+FN+1e-8)
    f1w  = f1_score(y_true, y_pred, average='weighted') if cm.size==4 else 0.0
    return acc, sens, spec, f1, f1w, cm

def scan_best_threshold(y_true, y_prob):
    best_val, best_th, best_pack = -1.0, 0.5, None
    for th in np.linspace(0.05, 0.9, 171):
        acc, sens, spec, f1, f1w, cm = compute_metrics(y_true, y_prob, th)
        val = 0.5*acc + 0.5*sens
        if val > best_val:
            best_val, best_th, best_pack = val, th, (acc, sens, spec, f1, f1w, cm)
    return best_th, best_pack, best_val

# ---------------- Grad-CAM / Grad-CAM++ (hook last EfficientNet feature block) --------------
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
    cam_rgb = cmap(cam)[..., :3]
    spec_rgb = np.stack([s,s,s], axis=-1)
    out = (1-alpha)*spec_rgb + alpha*cam_rgb
    return np.clip(out, 0.0, 1.0)

def gradcam_sequence(model: nn.Module, x: torch.Tensor, target_class: int = 1) -> List[np.ndarray]:
    """
    x: (1,T,21,H,W)  → list of (H,W) CAM per timestep (Grad-CAM)
    """
    device = next(model.parameters()).device
    model.eval()
    tgt = model.backbone.features[-1]  # last EfficientNet block
    hk = _Hook(tgt)

    x = x.clone().to(device).float().requires_grad_(True)

    with cam_modes(model):  # <<< add this
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
    """
    Grad-CAM++ per timestep.
    """
    device = next(model.parameters()).device
    model.eval()
    tgt = model.backbone.features[-1]
    hk = _Hook(tgt)

    x = x.clone().to(device).float().requires_grad_(True)
    with cam_modes(model):  # <<< add this
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


def ig_sequence(model: nn.Module, X_sample: torch.Tensor, baseline: torch.Tensor = None, n_steps: int = 64):
    """
    Integrated Gradients over seizure probability (class 1).
    Returns (T,H,W): mean over channels per timestep.
    """
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

    # IMPORTANT: allow RNN backward; keep backbone eval-stable
    with cam_modes(model):
        attrs, _ = ig.attribute(
            X_sample, baselines=baseline, target=None, n_steps=n_steps, return_convergence_delta=True
        )
    vals = attrs[0].detach().cpu().numpy()  # (T,21,H,W)
    per_t = vals.mean(axis=1)                # (T,H,W)
    return per_t


# ---------------- SHAP (optional) ----------------
def shap_sequence(model: nn.Module, X_sample: torch.Tensor, background: torch.Tensor, nsamples: int = 100):
    """
    SHAP GradientExplainer with PyTorch backend (no TF).
    Returns per-timestep (H,W) maps averaged over channels.
    """
    import shap
    device = next(model.parameters()).device
    model.eval()

    class ProbWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            return torch.softmax(out, dim=1)[:, 1:2]  # (B,1)

    wrapped = ProbWrapper(model).to(device)

    # pass TENSORS to select the PyTorch backend
    bg = background.to(device).float()     # (B_bg, T, 21, H, W)
    xs = X_sample.to(device).float()       # (1, T, 21, H, W)

    # cuDNN LSTM backward requires train() on the temporal block
    with cam_modes(model):
        explainer = shap.GradientExplainer(wrapped, bg)
        vals = explainer.shap_values(xs, nsamples=nsamples)[0]  # (1, T, 21, H, W)

    vals = vals[0]               # (T, 21, H, W)
    per_t = vals.mean(axis=1)    # (T, H, W)
    return per_t

# ---------------- Visualization utilities ----------------
def save_confusion_matrix(cm: np.ndarray, out_png: str):
    plt.figure(figsize=(4.6,4.2))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xticks([0,1], ['Non-seizure','Seizure'], rotation=20)
    plt.yticks([0,1], ['Non-seizure','Seizure'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()

def save_grid(images: List[np.ndarray], titles: List[str], out_png: str, cols: int=3):
    n = len(images)
    rows = (n + cols - 1)//cols
    plt.figure(figsize=(3.2*cols, 3.2*rows))
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.set_title(titles[i], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close()

# ---------------- Optional LLM summary (Gemini) ----------------
def gemini_summary(metrics: Dict[str, Any],
                   notes: List[str],
                   api_key: str,
                   model_name: str = "gemini-2.0-flash",
                   temperature: float = 0.25) -> str:
    if not api_key:
        return "LLM summary skipped (no API key)."
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    prompt = f"""Summarize EEG spectrogram classification results and explain why so in a clinical-educator tone>.
Metrics:
- Accuracy: {metrics['acc']:.4f}
- Sensitivity: {metrics['sens']:.4f}
- Specificity: {metrics['spec']:.4f}
- F1: {metrics['f1']:.4f}
- Weighted F1: {metrics['f1w']:.4f}
- Best threshold: {metrics['thr']:.2f}
Observations:
{chr(10).join('- '+x for x in notes)}
Explain what Grad-CAM and Grad-CAM++ highlighted that contribute to the seizure clasification."""
    body = {"contents":[{"parts":[{"text":prompt}]}],
            "generationConfig":{"temperature":temperature}}
    r = requests.post(url, json=body, timeout=90); r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

# ---------------- Main pipeline ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: uses 70/30 split inside, returns test set for unseen evaluation
    (Xtr, ytr), (Xte, yte) = load_patient_split(PATIENT, SPEC_DIR, seq_len=SEQ_LEN, augment=False)

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
    thr, (acc, sens, spec, f1, f1w, cm), obj = scan_best_threshold(y_true, y_prob)

    # Save confusion matrix
    cm_png = os.path.join(OUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(cm, cm_png)

    # --- enforce exactly 2 seizure + 2 non-seizure picks from TEST (if available) ---
    pos_idx = np.where(y_true == 1)[0].tolist()
    neg_idx = np.where(y_true == 0)[0].tolist()

    # deterministic order (keep as-is); take first 2 of each
    pos_sel = pos_idx[:2]
    neg_sel = neg_idx[:2]

    # fallback if one class has <2 (top up from the other class)
    if len(pos_sel) < 2 and len(neg_idx) > 0:
        need = 2 - len(pos_sel)
        pos_sel += neg_idx[:need]
    if len(neg_sel) < 2 and len(pos_idx) > 0:
        need = 2 - len(neg_sel)
        neg_sel += pos_idx[:need]

    # assemble up to 4 unique indices, prefer pos first then neg
    picks = []
    seen = set()
    for idx in pos_sel + neg_sel:
        if idx not in seen:
            seen.add(idx)
            picks.append(idx)
        if len(picks) == 4:
            break

    # Explainability per pick
    notes = []
    shap_files_per_sample = {}  # NEW: idx -> [list of shap file basenames]
    for idx in picks:
        xs = Xte[idx:idx+1].to(device).float()           # (1,T,21,H,W)
        T, H, W = xs.shape[1], xs.shape[-2], xs.shape[-1]
        base = xs[0, -1].detach().cpu().numpy().mean(axis=0)  # last step, mean over channels

        # Grad-CAM
        gcams = gradcam_sequence(model, xs, target_class=1)
        gcam_imgs = [_overlay(base, g) for g in gcams]
        gcam_png = os.path.join(OUT_DIR, f"sample{idx}_gradcam.png")
        save_grid(gcam_imgs, [f"t={t}" for t in range(T)], gcam_png)

        # Grad-CAM++
        gcpps = gradcampp_sequence(model, xs, target_class=1)
        gcpp_imgs = [_overlay(base, g) for g in gcpps]
        gcpp_png = os.path.join(OUT_DIR, f"sample{idx}_gradcampp.png")
        save_grid(gcpp_imgs, [f"t={t}" for t in range(T)], gcpp_png)

        
        # --- Integrated Gradients (Captum), per timestep ---
        ig_pngs = []
        try:
            per_t_maps = ig_sequence(model, xs, baseline=None, n_steps=64)  # (T,H,W)
            for t in range(per_t_maps.shape[0]):
                plt.figure(figsize=(3.2,3))
                plt.imshow(per_t_maps[t], cmap="seismic")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"IG t={t} (mean channels)")
                outp = os.path.join(OUT_DIR, f"sample{idx}_ig_t{t}.png")
                plt.tight_layout(); plt.savefig(outp, dpi=140); plt.close()
                ig_pngs.append(os.path.basename(outp))
        except Exception as e:
                print("IG attribution skipped:", e)

        
        
        
        shap_pngs = []
        try:
            bg_n = min(8, len(Xtr))
            bg = Xtr[:bg_n]  # torch tensor
            shap_maps = shap_sequence(model, xs, bg, nsamples=100)  # (T,H,W)
            
            with cam_modes(model):
                shap_maps = shap_sequence(model, xs, bg, nsamples=100)  # (T,H,W)
            
            for t in range(shap_maps.shape[0]):
                plt.figure(figsize=(3.2,3))
                plt.imshow(shap_maps[t], cmap="seismic")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"SHAP t={t} (mean channels)")
                outp = os.path.join(OUT_DIR, f"sample{idx}_shap_t{t}.png")
                plt.tight_layout(); plt.savefig(outp, dpi=140); plt.close()
                shap_pngs.append(outp)
        except Exception as e:
            print("SHAP attribution skipped:", e)
        shap_files_per_sample[idx] = shap_pngs

        p_hat = y_prob[idx]; pred = int(p_hat >= thr)
        files_list = [os.path.basename(gcam_png), os.path.basename(gcpp_png)] + shap_files_per_sample[idx] + ig_pngs
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
        f.write("## Examples (Grad-CAM / Grad-CAM++)\n")
        for idx in picks:
            f.write(f"\n**Sample {idx}**\n\n")
            f.write(f"![](sample{idx}_gradcam.png)\n\n")
            f.write(f"![](sample{idx}_gradcampp.png)\n\n")

            for shp in shap_files_per_sample.get(idx, []):
                f.write(f"![]({shp})\n\n")

            for igf in ig_pngs:  # you may want to store per-sample if you prefer
                f.write(f"![]({igf})\n\n")

        metrics = {"acc":acc, "sens":sens, "spec":spec, "f1":f1, "f1w":f1w, "thr":thr}
        try:
            summary = gemini_summary(metrics, notes, GEMINI_API_KEY) if GEMINI_API_KEY else "LLM summary skipped."
        except Exception as e:
            summary = f"LLM summary error: {e}"
        f.write("## LLM Summary\n")
        f.write(summary + "\n")
        f.write("\n## Notes\n")
        for s in notes: f.write("- " + s + "\n")

    print("Report folder:", OUT_DIR)
    print("Markdown:", md)

if __name__ == "__main__":
    main()
