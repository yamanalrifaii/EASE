import os, json, torch, numpy as np
import matplotlib.pyplot as plt
from architecture_spectrograms import Net
from loader_spectrograms import load_patient_split
from xai_gradcam import GradCAM, overlay_heatmap   # from the snippet earlier
from report_writer import draft_report             # Gemini report function

# ----------------------------
# SETTINGS
# ----------------------------
PATIENT = "chb01"
SPEC_DIR = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
CKPT = r"E:\EEG\spectrograms\models_optuna_gemini_boost\per_patient_results\chb01\chb01_trial0.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# OUTPUT FOLDER (your request)
# ----------------------------
OUT_DIR = r"E:\EEG\xai_llm\chb01"
os.makedirs(OUT_DIR, exist_ok=True)      # create folder if missing

# ----------------------------
# LOAD DATA (your loader v2) :contentReference[oaicite:2]{index=2}
# ----------------------------
(_, _), (X_test, y_test) = load_patient_split(PATIENT, SPEC_DIR)
print("Loaded test set:", X_test.shape, "labels:", dict(zip(*np.unique(y_test, return_counts=True))))

# take a single test spectrogram (e.g., first seizure)
idx = int((y_test == 1).nonzero(as_tuple=True)[0][0])
spec = X_test[idx:idx+1].to(DEVICE)
print("Selected sample idx:", idx)

# ----------------------------
# LOAD TRAINED MODEL :contentReference[oaicite:3]{index=3}
# ----------------------------
model = Net(n_classes=2).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ----------------------------
# PREDICT
# ----------------------------
with torch.no_grad():
    logits = model(spec)
    probs = torch.softmax(logits, dim=1)[0,1].item()
print(f"Seizure probability: {probs:.3f}")

# ----------------------------
# GRAD-CAM EXPLAINABILITY
# ----------------------------
cam = GradCAM(model, target_layer="features.6").generate(spec, class_idx=1)
np.save("chb01_gradcam.npy", cam)

# spectrogram image
spec_path = os.path.join(OUT_DIR, "spectrogram.png")
plt.imsave(spec_path, spec[0,0].detach().cpu().numpy(), cmap="magma")

# grad-cam overlay image (using your overlay_heatmap from xai_gradcam.py)
overlay = overlay_heatmap(spec[0,0].detach().cpu().numpy(), cam, alpha=0.45)
gradcam_path = os.path.join(OUT_DIR, "gradcam_overlay.png")
overlay.save(gradcam_path)

# raw gradcam array
np.save(os.path.join(OUT_DIR, "chb01_gradcam.npy"), cam)


# ----------------------------
#  Quantify Grad-CAM attention  (add this before cnn_json)
# ----------------------------
import numpy as np

# cam is your [H, W] heatmap array (values 0..1)
hot_ratio = float((cam > 0.7).mean() * 100)  # % of spectrogram highlighted
hot_time_region = np.argmax(cam.mean(axis=0))  # column index with max activation
hot_freq_region = np.argmax(cam.mean(axis=1))  # row index with max activation

xai_summary = {
    "hot_ratio_percent": round(hot_ratio, 1),
    "max_activation_time_index": int(hot_time_region),
    "max_activation_freq_index": int(hot_freq_region),
    "interpretation": (
        "Grad-CAM highlighted ~{:.1f}% of the spectrogram, "
        "with strongest activation near time column {} and frequency row {}."
    ).format(hot_ratio, hot_time_region, hot_freq_region)
}


# ----------------------------
# BUILD JSON FINDINGS
# ----------------------------
cnn_json = {
    "recording_id": f"{PATIENT}_demo",
    "global_confidence": float(probs),
    "seizure_events": [{"start_sec": 0, "end_sec": 4, "confidence": float(probs)}],
    "xai_method": "Grad-CAM",
    "xai_summary": xai_summary
}

# pass the OUT_DIR paths to the report writer
from report_writer import draft_report
report = draft_report(spec_path, gradcam_path, cnn_json)

# save report JSON in OUT_DIR
import json
with open(os.path.join(OUT_DIR, "chb01_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("Saved to:", OUT_DIR)
