# =============================================================
# trainer.py  —  EEGWaveNet trainer (balanced batches, GPU-safe)
# =============================================================
import time, gc
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

class trainer:
    def __init__(self, Model, Train_set):
        self.Model = Model
        self.X_train, self.y_train = Train_set  # will be replaced per file in LOO
        self.loss_func = None
        self.compiled = False

    def compile(self, lr=1e-3, pos_weight=None):
        self.optimizer = Adam(self.Model.parameters(), lr=lr)
        # Optional class weighting (pos_weight > 1 to up-weight seizures)
        if pos_weight is not None:
            w = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
            if torch.cuda.is_available():
                w = w.cuda()
            self.loss_func = nn.CrossEntropyLoss(weight=w)
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.compiled = True

    def _balanced_indices(self):
        """
        Build balanced 0/1 indices for current self.y_train.
        If only one class is present, fall back to using all samples.
        """
        if isinstance(self.y_train, torch.Tensor):
            y_np = self.y_train.cpu().numpy()
        else:
            y_np = np.asarray(self.y_train)

        pos_idx = np.where(y_np == 1)[0]
        neg_idx = np.where(y_np == 0)[0]

        if len(pos_idx) > 0 and len(neg_idx) > 0:
            n = min(len(pos_idx), len(neg_idx))
            # Cap to avoid insanely large balanced sets per file
            max_total = 4096
            n = min(n, max_total // 2)

            sel_pos = np.random.choice(pos_idx, n, replace=False)
            sel_neg = np.random.choice(neg_idx, n, replace=False)
            idx = np.concatenate([sel_pos, sel_neg])
            np.random.shuffle(idx)
            print(f"Epoch: balanced 0={n} 1={n}")
        else:
            # Only one class present (all 0 or all 1) – use everything
            idx = np.arange(len(y_np))
            only = int(y_np[0]) if len(y_np) > 0 else -1
            print(f"Epoch: only class {only} present, using all {len(y_np)} samples")

        return idx

    def train(self, epochs=3, batch_size=512, directory="model.pt"):
        if not self.compiled:
            raise RuntimeError("Call compile() before train().")

        if torch.cuda.is_available():
            self.Model.cuda()

        for e in range(epochs):
            idx = self._balanced_indices()
            X_epoch = self.X_train[idx]
            y_epoch = self.y_train[idx]

            dataset = TensorDataset(X_epoch, y_epoch)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            T0 = time.time()
            self.Model.train()
            running_loss = 0.0
            batches = 0

            for data, target in loader:
                data = data.float()
                target = target.long()
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                out = self.Model(data)
                loss = self.loss_func(out, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batches += 1

            train_loss = running_loss / max(batches, 1)
            print(f"Epoch {e:03d} | Train {train_loss:.5f} | {time.time() - T0:.1f}s")

            # Save after each epoch (optional)
            torch.save(self.Model.state_dict(), directory)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("✅ Training finished. Final model saved to", directory)

    def predict_proba(self, X, batch_size=512):
        """Return seizure probability for each sample."""
        self.Model.eval()
        preds = []
        loader = DataLoader(torch.tensor(X, dtype=torch.float32),
                            batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for data in loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                logits = self.Model(data)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)
                preds.extend(probs.cpu().numpy())
        return np.array(preds)

    def predict(self, X, batch_size=512, threshold=0.5):
        """Return hard class labels 0/1 with given threshold."""
        probs = self.predict_proba(X, batch_size=batch_size)
        return (probs > threshold).astype(int)