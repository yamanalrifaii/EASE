# =============================================================
# dataset_loader_spectrogram.py — patient-dependent loader (21ch, v2)
# =============================================================
import os, glob, numpy as np, torch
from sklearn.model_selection import train_test_split

def load_patient_split(patient_id, spec_dir, test_size=0.3):
    """
    Loads all *_spectrogram.npy and *_labels.npy for one patient,
    applies per-sample standardization, mild augmentation, and split.
    """
    spec_files = sorted(glob.glob(os.path.join(spec_dir, f"{patient_id}_*_spectrogram.npy")))
    label_files = sorted(glob.glob(os.path.join(spec_dir, f"{patient_id}_*_labels.npy")))

    if not spec_files or not label_files:
        raise FileNotFoundError(f"No spectrogram or label files found for {patient_id} in {spec_dir}")

    Xs, ys = [], []
    for spec_f, lbl_f in zip(spec_files, label_files):
        X = np.load(spec_f, allow_pickle=True)   # (segments, H, W, 21)
        y = np.load(lbl_f, allow_pickle=True).astype(np.int64)

        if len(X) != len(y):
            print(f"⚠️ Skipping {spec_f}: mismatch ({len(X)} vs {len(y)})")
            continue

        X = X.astype(np.float32)
        mean = np.mean(X, axis=(1, 2, 3), keepdims=True)
        std  = np.std(X,  axis=(1, 2, 3), keepdims=True) + 1e-8
        X = (X - mean) / std

        # --- Mild augmentation (applied randomly) ---
        for i in range(len(X)):
            if np.random.rand() < 0.3:
                X[i] = X[i] + np.random.normal(0, 0.02, X[i].shape)  # Gaussian noise
            if np.random.rand() < 0.3:
                X[i] = np.flip(X[i], axis=1).copy()  # Horizontal flip (time axis)

        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # --- Convert to PyTorch tensors (NCHW) ---
    X_train = torch.tensor(np.transpose(X_train, (0, 3, 1, 2)), dtype=torch.float32)
    X_test  = torch.tensor(np.transpose(X_test,  (0, 3, 1, 2)), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    print(f"\n{patient_id}: train {len(y_train)}  test {len(y_test)}")
    print("train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("test  labels:", dict(zip(*np.unique(y_test,  return_counts=True))))
    print("Sample batch mean/std:", float(X_train.mean()), float(X_train.std()))

    return (X_train, y_train), (X_test, y_test)