# =============================================================
# dataset_spectrogram_stream.py
# Streams EEG spectrograms directly from disk (RAM-safe)
# =============================================================
import os, glob, numpy as np, torch
from torch.utils.data import Dataset
import gc, random

class EEGSpectrogramStreamDataset(Dataset):
    def __init__(self, spec_dir, seq_len=3, patient_subset=None, augment=False):
        """
        spec_dir: folder with *_spectrogram.npy and *_labels.npy
        seq_len : number of consecutive spectrograms per sample
        patient_subset: list like ['chb01','chb02'] to limit patients
        augment: whether to apply mild on-the-fly augmentations
        """
        self.seq_len = seq_len
        self.augment = augment

        self.spec_label_pairs = []
        all_files = sorted(glob.glob(os.path.join(spec_dir, "*_spectrogram.npy")))
        if patient_subset:
            all_files = [f for f in all_files
                         if os.path.basename(f).split("_")[0] in patient_subset]

        for spec_f in all_files:
            lbl_f = spec_f.replace("_spectrogram.npy", "_labels.npy")
            if not os.path.exists(lbl_f): continue
            self.spec_label_pairs.append((spec_f, lbl_f))

        print(f"🧠 Streaming dataset initialized with {len(self.spec_label_pairs)} file pairs.")

        # Precompute (file_id, index_in_file) for all valid sequences
        self.index_map = []
        for file_id, (sf, lf) in enumerate(self.spec_label_pairs):
            try:
                y = np.load(lf, mmap_mode="r", allow_pickle=True)
                n = len(y)
                for i in range(n - seq_len + 1):
                    self.index_map.append((file_id, i))
            except Exception as e:
                print(f"⚠️ Skip {sf}: {e}")
        print(f"✅ Total sequences available: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_id, start = self.index_map[idx]
        spec_f, lbl_f = self.spec_label_pairs[file_id]

        X_mm = np.load(spec_f, mmap_mode="r", allow_pickle=True)
        y_mm = np.load(lbl_f, mmap_mode="r", allow_pickle=True).astype(np.int64)

        frames = []
        for j in range(self.seq_len):
            f = np.array(X_mm[start + j], dtype=np.float32)
            f = (f - f.mean()) / (f.std() + 1e-8)
            if self.augment and random.random() < 0.4:
                f = np.flip(f, axis=1).copy()
            frames.append(f)

        seq = np.stack(frames, axis=0)
        label = int(y_mm[start + self.seq_len - 1])
        del X_mm, y_mm, frames, f
        gc.collect()

        # (seq_len, 21, H, W)
        seq = torch.tensor(np.transpose(seq, (0,3,1,2)), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return seq, label