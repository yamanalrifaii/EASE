# =============================================================
# architecture.py  —  EEGWaveNet-like 1D CNN for CHB-MIT
# =============================================================
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, n_chans, n_classes):
        super(Net, self).__init__()

        # --- Multiscale temporal depthwise convs (like EEGWaveNet) ---
        # Input: (B, n_chans, 1024)
        self.temp_conv = nn.ModuleList([
            nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
            for _ in range(6)
        ])
        # ts[0] length: 1024
        # ts[1] length: 512
        # ts[2] length: 256
        # ts[3] length: 128
        # ts[4] length: 64
        # ts[5] length: 32
        # ts[6] length: 16

        def chblock(in_ch):
            # small channel-wise feature extractor
            return nn.Sequential(
                nn.Conv1d(in_ch, 32, kernel_size=1),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.01),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.01),
            )

        # five multiscale branches (using ts[1]..ts[5])
        self.chpools = nn.ModuleList([chblock(n_chans) for _ in range(5)])

        # classifier on concatenated multiscale features (5 * 32 = 160-dim)
        self.classifier = nn.Sequential(
            nn.Linear(5 * 32, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x: (B, C=21, T=1024)
        ts = [x]
        for conv in self.temp_conv:
            ts.append(conv(ts[-1]))  # each step halves temporal length

        # multiscale: use ts[1]..ts[5]
        ws = []
        for i in range(5):
            h = self.chpools[i](ts[i+1])      # (B, 32, L_i)
            h = h.mean(-1)                    # global average over time -> (B, 32)
            ws.append(h)

        feat = torch.cat(ws, dim=1)          # (B, 5*32)
        logits = self.classifier(feat)       # (B, n_classes)
        # IMPORTANT: return raw logits (NO log_softmax here!)
        return logits