# =============================================================
# generate_spectrograms_21ch_stft_fixed.py
# Generates 21-channel STFT spectrograms for CHB-MIT EEG segments
# Works for both dict-type and plain ndarray segments
# =============================================================

import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.signal import spectrogram
import random

# =============================================================
# CONFIGURATION
# =============================================================
INPUT_DIR  = r"E:\EEG\chbmit\segments_db4"                 # segmented EEG signals
LABEL_DIR  = r"E:\EEG\chbmit\manual_labels_db4"            # matching labels
OUTPUT_DIR = r"E:\EEG\chbmit\segment_spectrograms_21ch_stft_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 256
NPERSEG = 256
NOVERLAP = 128
OUT_H, OUT_W = 128, 256
RATIO = 2  # Non-seizure : seizure ratio (2:1)

# =============================================================
# DEFAULT CHANNEL ORDER (used when ch_names missing)
# =============================================================
CHBMIT_DEFAULT_CH_NAMES = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8", "FP1-F3", "FP2-F4"
]

# =============================================================
# TARGET 21 CHANNELS (we’ll search for these)
# =============================================================
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8", "FP1-F3", "FP2-F4"
]

# =============================================================
# HELPER: Compute spectrogram
# =============================================================
def make_spectrogram(signal):
    """
    Compute log-scaled, normalized STFT spectrogram for one EEG channel.
    Returns an array of shape (OUT_H, OUT_W) in range [0, 1].
    """
    # 1. Convert to microvolts
    signal = signal.astype(np.float32) * 1e6

    # 2. Remove DC drift (important for low-amplitude EEG)
    signal -= np.mean(signal)

    # 3. Compute STFT power spectral density
    f, t, Sxx = spectrogram(
        signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
        scaling='density', mode='magnitude'
    )

    # 4. Clip to avoid underflow and log-scale
    Sxx = np.maximum(Sxx, 1e-12)
    Sxx = np.log10(Sxx)

    # 5. Normalize globally per spectrogram
    Sxx_min, Sxx_max = np.percentile(Sxx, 1), np.percentile(Sxx, 99)
    Sxx = np.clip((Sxx - Sxx_min) / (Sxx_max - Sxx_min + 1e-12), 0, 1)

    # 6. Resize for consistent CNN input
    Sxx = cv2.resize(Sxx, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

    return Sxx.astype(np.float32)
# =============================================================
# MAIN LOOP
# =============================================================
patients = sorted({f.split('_')[0] for f in os.listdir(LABEL_DIR)})

for patient in patients:
    print(f"\n🧠 Processing {patient}...")

    label_files = [f for f in os.listdir(LABEL_DIR)
                   if f.startswith(patient) and f.endswith('.npy')]

    for label_file in label_files:
        base_name = label_file.replace("_labels.npy", "")
        label_path = os.path.join(LABEL_DIR, label_file)
        segment_file = os.path.join(INPUT_DIR, base_name + "_segments.npy")

        if not os.path.exists(segment_file):
            print(f"⚠️ Missing segments for {base_name}, skipping.")
            continue

        # --- Load labels and indices ---
        labels = np.load(label_path)
        seiz_idx = np.where(labels == 1)[0].tolist()
        non_idx  = np.where(labels == 0)[0].tolist()

        if len(seiz_idx) == 0:
            print(f"⚠️ No seizures in {base_name}, skipping.")
            continue

        n_keep = min(len(non_idx), len(seiz_idx) * RATIO)
        non_idx = random.sample(non_idx, n_keep)
        selected_idx = sorted(seiz_idx + non_idx)
        print(f"→ Using {len(seiz_idx)} seizure + {len(non_idx)} non-seizure segments")

        # --- Load EEG segments ---
        segments = np.load(segment_file, allow_pickle=True)
        spectrograms, labels_used = [], []

        for idx in tqdm(selected_idx, desc=base_name):
            seg_data = segments[idx]

            # Unwrap nested formats
            if isinstance(seg_data, np.ndarray):
                if seg_data.size == 1:
                    seg_data = seg_data.item()
                elif isinstance(seg_data[0], dict):
                    seg_data = seg_data[0]

            # Get data and channel names
            if isinstance(seg_data, dict):
                eeg = seg_data["data"]
                ch_names = [ch.upper().strip() for ch in seg_data["ch_names"]]
            elif isinstance(seg_data, np.ndarray):
                eeg = seg_data
                ch_names = [ch.upper() for ch in CHBMIT_DEFAULT_CH_NAMES[:eeg.shape[0]]]
            else:
                continue

            if eeg.shape[0] < 21:
                print(f"⚠️ Only {eeg.shape[0]} channels, skipping segment {idx}")
                continue

            # --- Compute spectrograms for 21 channels ---
            spectros = []
            for ch_name in TARGET_CHANNELS:
                if ch_name.upper() in ch_names:
                    ch_idx = ch_names.index(ch_name.upper())
                    spec = make_spectrogram(eeg[ch_idx])
                else:
                    spec = np.zeros((OUT_H, OUT_W), dtype=np.float32)
                spectros.append(spec)

            stacked = np.stack(spectros, axis=-1)  # (128, 256, 21)
            spectrograms.append(stacked)
            labels_used.append(int(labels[idx]))

        # --- Save results ---
        if len(spectrograms) == 0:
            print(f"⚠️ No spectrograms created for {base_name}.")
            continue

        spectrograms = np.array(spectrograms, dtype=np.float32)
        labels_used = np.array(labels_used, dtype=np.uint8)

        out_spec = os.path.join(OUTPUT_DIR, f"{base_name}_spectrogram.npy")
        out_label = os.path.join(OUTPUT_DIR, f"{base_name}_labels.npy")

        np.save(out_spec, spectrograms)
        np.save(out_label, labels_used)

        print(f"✅ Saved {spectrograms.shape[0]} samples (shape {spectrograms.shape[1:]}, "
              f"mean={spectrograms.mean():.4f}, std={spectrograms.std():.4f}) → {out_spec}")