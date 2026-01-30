# =============================================================
# trainer.py  —  Improved EEG Spectrogram Trainer (v2)
# =============================================================
import torch, numpy as np, time, gc
from torch import nn
from torch.optim import Adam

class trainer:
    def __init__(self, Model, Train_set):
        self.Model = Model
        self.X_train, self.y_train = Train_set
        # Label smoothing helps prevent overconfidence
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).cuda())
        self.compiled = False

    def compile(self, lr=1e-3):
        # Add weight decay for better generalization
        self.optimizer = Adam(self.Model.parameters(), lr=lr, weight_decay=1e-4)

        # Add scheduler: reduces LR when loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.compiled = True

    def train(self, epochs=70, batch_size=64, directory="model.pt"):
        if torch.cuda.is_available():
            self.Model.cuda()

        for e in range(epochs):
            T0 = time.time()
            self.Model.train()

            # --- 2:1 balancing ---
            y_np = self.y_train.cpu().numpy()
            seiz = np.where(y_np == 1)[0]
            norm = np.where(y_np == 0)[0]
            if len(seiz) == 0:
                print(f"Epoch {e}: no seizure samples, skip"); continue
            np.random.shuffle(norm)
            ratio = 1  # 2:1 non-seizure:seizure
            norm = norm[:min(len(norm), len(seiz)*ratio)]
            idx = np.concatenate([seiz, norm]); np.random.shuffle(idx)
            Xb, yb = self.X_train[idx], self.y_train[idx]
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
                loss = self.loss_func(self.Model(data.float()), target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach().cpu())

            train_loss = torch.mean(torch.stack(losses))
            print(f"Epoch {e:03d} | Train {train_loss:.5f} | {time.time()-T0:.1f}s")

            # Update scheduler
            self.scheduler.step(train_loss)

            # Save model checkpoint
            torch.save(self.Model.state_dict(), directory)
            gc.collect(); torch.cuda.empty_cache()

        print("✅ Training finished. Final model saved to", directory)

    def predict(self, X):
        self.Model.eval(); preds=[]
        loader = torch.utils.data.DataLoader(X, batch_size=512, shuffle=False)
        with torch.no_grad():
            for data in loader:
                if torch.cuda.is_available(): data=data.cuda()
                preds.extend(torch.argmax(self.Model(data.float()), dim=1).cpu().numpy())
        return np.array(preds)