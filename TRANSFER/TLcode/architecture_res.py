# =============================================================
# architecture_transfer_resnet34_cnn.py
# ResNet-34 for EEG Spectrograms (21 channels, no LSTM)
# =============================================================
import torch
import torch.nn as nn
from torchvision import models

class EEGSpectrogramResNet34_CNN(nn.Module):
    def __init__(self, n_classes=2, fine_tune=True):
        super().__init__()

        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Adapt first conv layer: 3 → 21 channels
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels=21,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.repeat(1, 7, 1, 1) / 7.0)
        base.conv1 = new_conv

        # Replace classifier with Dropout + Linear
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, n_classes)
        )
        self.model = base

        # Fine-tune policy
        if fine_tune:
            for p in self.model.parameters():
                p.requires_grad = False
            for name, p in self.model.named_parameters():
                if (
                    name.startswith("layer2")
                    or name.startswith("layer3")
                    or name.startswith("layer4")
                    or "bn" in name
                ):
                    p.requires_grad = True

    def forward(self, x):
        return self.model(x)

Net = EEGSpectrogramResNet34_CNN