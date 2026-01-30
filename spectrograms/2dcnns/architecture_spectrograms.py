# =============================================================
# cnn_spectrogram_21ch.py — 2D CNN for 21-channel spectrograms (v2)
# =============================================================
import torch
from torch import nn

class CNN_Spectrogram_21ch(nn.Module):
    def __init__(self, n_classes=2):
        super(CNN_Spectrogram_21ch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(21, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # ↑ from 0.5 for better regularization
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

Net = CNN_Spectrogram_21ch