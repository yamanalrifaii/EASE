# =============================================================
# trainer.py — Final Advanced EEG Spectrogram Trainer (v6)
# =============================================================
import torch, numpy as np, time, gc
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# -------------------------------------------------------------
# Focal Loss (balanced for seizure detection)
# -------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 1.5], gamma=1.5):
        super().__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=self.alpha)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        loss = ((1 - p) ** self.gamma) * ce_loss
        return loss.mean()


# -------------------------------------------------------------
# Trainer class
# -------------------------------------------------------------
class trainer:
    def __init__(self, Model, Train_set):
        self.Model = Model
        self.X_train, self.y_train = Train_set
        self.loss_func = FocalLoss(alpha=[1.0, 1.5], gamma=1.5)
        self.compiled = False

    # ----------------------------------------------------------
    # Differential LR + warm-up + cosine decay
    # ----------------------------------------------------------
    def compile(self, lr=1e-4):
        params = []
        for name, param in self.Model.named_parameters():
            if not param.requires_grad:
                continue
            lr_factor = 0.3 if not name.startswith("model.fc") else 1.0
            params.append({"params": param, "lr": lr * lr_factor})

        self.optimizer = Adam(params, weight_decay=1e-4)

        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=5)
        cosine = CosineAnnealingLR(self.optimizer, T_max=150, eta_min=lr * 0.01)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[5])

        self.compiled = True

    # ----------------------------------------------------------
    # Gradual Unfreezing
    # ----------------------------------------------------------
    def gradual_unfreeze(self, epoch):
        if epoch == 30:
            print("🔓 Unfreezing layer3...")
            for name, p in self.Model.named_parameters():
                if "layer3" in name:
                    p.requires_grad = True
        elif epoch == 60:
            print("🔓 Unfreezing layer2...")
            for name, p in self.Model.named_parameters():
                if "layer2" in name:
                    p.requires_grad = True
        elif epoch == 100:
            print("🔓 Unfreezing layer1 and conv1...")
            for name, p in self.Model.named_parameters():
                if "layer1" in name or "conv1" in name:
                    p.requires_grad = True

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    def train(self, epochs=150, batch_size=64, directory="model.pt", val_set=None):
        if torch.cuda.is_available():
            self.Model.cuda()

        X_val, y_val = val_set if val_set else (None, None)

        for e in range(epochs):
            T0 = time.time()
            self.Model.train()
            self.gradual_unfreeze(e)

            # --- 1:1 balancing (light oversampling of seizures) ---
            y = self.y_train.cpu().numpy()
            idx_pos = np.where(y == 1)[0]
            idx_neg = np.where(y == 0)[0]
            print(f"Epoch {e}: balanced 0={(yb==0).sum().item()} 1={(yb==1).sum().item()}")

            loader = torch.utils.data.DataLoader(
                [[Xb[i], yb[i]] for i in range(len(yb))],
                batch_size=batch_size, shuffle=True
            )

            losses = []
            for data, target in loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                self.optimizer.zero_grad()
                data = data.squeeze(1)
                output = self.Model(data.float())
                loss = self.loss_func(output, target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach().cpu())

            train_loss = torch.mean(torch.stack(losses))
            self.scheduler.step()
            print(f"Epoch {e:03d} | Train {train_loss:.5f} | {time.time()-T0:.1f}s")

            torch.save(self.Model.state_dict(), directory)
            gc.collect(); torch.cuda.empty_cache()

        print("✅ Training finished. Final model saved to", directory)

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    def predict(self, X):
        self.Model.eval(); preds=[]
        loader = torch.utils.data.DataLoader(X, batch_size=512, shuffle=False)
        with torch.no_grad():
            for data in loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                data = data.squeeze(1)

                preds.extend(torch.argmax(self.Model(data.float()), dim=1).cpu().numpy())
        return np.array(preds)