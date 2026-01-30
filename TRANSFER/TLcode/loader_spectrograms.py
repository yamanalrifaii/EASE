# =============================================================
# loader_spectrograms.py — EEG Spectrogram Loader (with sequence support)
# =============================================================
import os, glob, numpy as np, torch
from sklearn.model_selection import train_test_split

# =============================================================
# Main loader
# =============================================================
def load_patient_split(patient_id, spec_dir, test_size=0.3, augment=True, seq_len=3):
    """
    Loads spectrograms for one patient and returns training / testing tensors.

    For ResNet-LSTM: stacks `seq_len` consecutive spectrograms per sample
    -> output shape (B, seq_len, C, H, W)
    """
    spec_files = sorted(glob.glob(os.path.join(spec_dir, f"{patient_id}_*_spectrogram.npy")))
    label_files = sorted(glob.glob(os.path.join(spec_dir, f"{patient_id}_*_labels.npy")))

    if not spec_files or not label_files:
        raise FileNotFoundError(f"No spectrogram/label files found for {patient_id}")

    Xs, ys = [], []
    for spec_f, lbl_f in zip(spec_files, label_files):
        X = np.load(spec_f, allow_pickle=True)   # (segments, H, W, 21)
        y = np.load(lbl_f, allow_pickle=True).astype(np.int64)
        if len(X) != len(y):
            print(f"⚠️ Skipping {spec_f}: mismatch ({len(X)} vs {len(y)})")
            continue

        # --- Per-sample normalization ---
        X = X.astype(np.float32)
        mean = np.mean(X, axis=(1, 2, 3), keepdims=True)
        std = np.std(X, axis=(1, 2, 3), keepdims=True) + 1e-8
        X = (X - mean) / std

        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # --- Create temporal sequences ---
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(np.stack([X[j] for j in range(i, i + seq_len)], axis=0))  # (seq_len, H, W, 21)
        y_seq.append(y[i + seq_len - 1])  # label of last frame
    X, y = np.array(X_seq), np.array(y_seq)

    # --- Split into train / test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # --- Apply augmentations only on training ---
    if augment:
        X_train = apply_augmentations(X_train)

    # --- Convert to PyTorch tensors (B, seq_len, C, H, W) ---
    X_train = torch.tensor(np.transpose(X_train, (0, 1, 4, 2, 3)), dtype=torch.float32)
    X_test  = torch.tensor(np.transpose(X_test,  (0, 1, 4, 2, 3)), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    print(f"\n{patient_id}: train {len(y_train)}  test {len(y_test)}")
    print("train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("test  labels:", dict(zip(*np.unique(y_test,  return_counts=True))))
    print("Sample batch mean/std:", float(X_train.mean()), float(X_train.std()))

    return (X_train, y_train), (X_test, y_test)

# =============================================================
# Augmentations — applied to each frame within a sequence
# =============================================================
def apply_augmentations(X):
    """Applies mild EEG-style augmentations to each spectrogram frame."""
    for i in range(len(X)):
        for t in range(X.shape[1]):  # loop over sequence length
            frame = X[i, t]
            if np.random.rand() < 0.4:
                frame = frame + np.random.normal(0, 0.02, frame.shape)
            if np.random.rand() < 0.4:
                frame = np.flip(frame, axis=1).copy()  # time flip
            if np.random.rand() < 0.4:
                shift = np.random.randint(-10, 10)
                frame = np.roll(frame, shift, axis=1)
            if np.random.rand() < 0.4:
                f_mask = np.random.randint(5, 15)
                f_start = np.random.randint(0, frame.shape[0] - f_mask)
                frame[f_start:f_start+f_mask, :] = 0
            if np.random.rand() < 0.4:
                frame = frame * np.random.uniform(0.9, 1.1)
            X[i, t] = frame
    return X