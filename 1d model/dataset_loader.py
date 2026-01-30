# =============================================================
# dataset_loader.py  —  FINAL (Segment-level stratified split, patient-dependent)
# =============================================================
import os, glob, numpy as np, torch
from sklearn.model_selection import train_test_split

def load_patient_split(patient_id, seg_dir, label_dir,
                       n_channels=21, test_size=0.3):
    """
    Loads all *_segments.npy + *_labels.npy for a single patient,
    concatenates them, normalizes each segment, and performs
    a stratified train/test split to ensure both contain seizures.
    """

    # --- locate all patient segment and label files ---
    seg_files = sorted(glob.glob(os.path.join(seg_dir,  f"{patient_id}_*_segments.npy")))
    lbl_files = sorted(glob.glob(os.path.join(label_dir, f"{patient_id}_*_labels.npy")))

    if not seg_files or not lbl_files:
        raise FileNotFoundError(f"No segment or label files found for {patient_id}.")

    Xs, ys = [], []

    # --- load and preprocess each file ---
    for seg_f, lbl_f in zip(seg_files, lbl_files):
        if not os.path.exists(seg_f) or not os.path.exists(lbl_f):
            print(f"⚠️ Missing file pair for {patient_id}: {os.path.basename(seg_f)}")
            continue

        X = np.load(seg_f, allow_pickle=True)        # (segments, channels, time)
        y = np.load(lbl_f, allow_pickle=True).astype(np.int64)

        if y.ndim > 1:
            y = np.squeeze(y)
        if len(X) != len(y):
            print(f"⚠️ Skipping {seg_f}: segment-label mismatch ({len(X)} vs {len(y)})")
            continue

        # --- keep first 21 channels ---
        X = X[:, :n_channels, :].astype(np.float32)

        # --- normalize each segment independently ---
        mean = np.mean(X, axis=2, keepdims=True)
        std  = np.std(X, axis=2, keepdims=True) + 1e-8
        X = (X - mean) / std

        Xs.append(X)
        ys.append(y)

    if not Xs:
        raise ValueError(f"No valid data loaded for {patient_id}.")

    # --- concatenate all segments for the patient ---
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # --- stratified train/test split ---
    if len(np.unique(y)) == 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
    else:
        print(f"⚠️ Only one class found for {patient_id}, splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

    # --- log summary ---
    print(f"\n{patient_id}: train {len(y_train)}  test {len(y_test)}")
    print("train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("test  labels:", dict(zip(*np.unique(y_test,  return_counts=True))))

    # --- convert to torch tensors ---
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    return (X_train, y_train), (X_test, y_test)