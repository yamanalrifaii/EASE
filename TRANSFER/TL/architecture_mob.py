# =============================================================
# architecture_transfer_mobilenet_lstm.py
# MobileNetV2 + LSTM hybrid for EEG Spectrograms
# =============================================================
import torch
import torch.nn as nn
from torchvision import models


class EEGSpectrogramMobileNetV2_LSTM(nn.Module):
    def __init__(self, n_classes=2, seq_len=3, hidden_size=256, fine_tune=True):
        super().__init__()
        self.seq_len = seq_len

        # ---------------------------------------------------------
        # 1️⃣ Base: Pretrained MobileNetV2 backbone
        # ---------------------------------------------------------
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Adapt first conv layer to 21 channels
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=21,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Repeat ImageNet weights across 21 channels and average
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.repeat(1, 7, 1, 1) / 7.0  # 3 * 7 = 21
            )
        base.features[0][0] = new_conv

        # Remove original classifier to use as feature extractor
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Identity()

        self.backbone = base
        self.feature_dim = in_features

        # ---------------------------------------------------------
        # 2️⃣ LSTM head for temporal modeling
        # ---------------------------------------------------------
        self.temporal = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # ---------------------------------------------------------
        # 3️⃣ Classifier head
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(hidden_size, n_classes),
        )

        # ---------------------------------------------------------
        # 4️⃣ Fine-tuning policy
        # ---------------------------------------------------------
        if fine_tune:
            # freeze everything first
            for p in self.backbone.parameters():
                p.requires_grad = False

            # unfreeze later / higher layers + batchnorms
            for name, p in self.backbone.named_parameters():
                # MobileNetV2 blocks are in features[1:] etc.
                if (
                    name.startswith("features.12")
                    or name.startswith("features.13")
                    or name.startswith("features.14")
                    or name.startswith("features.15")
                    or name.startswith("features.16")
                    or name.startswith("features.17")
                    or "bn" in name
                ):
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

        # LSTM + classifier are always trainable
        for p in self.temporal.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    # -------------------------------------------------------------
    # x shape: (batch, seq_len, channels, H, W)
    # -------------------------------------------------------------
    def forward(self, x):
        B, T, C, H, W = x.shape
        # Merge batch and time to feed CNN
        x = x.view(B * T, C, H, W)          # (B*T, 21, H, W)

        feats = self.backbone(x)            # (B*T, feature_dim)
        feats = feats.view(B, T, -1)        # (B, T, feature_dim)

        out, _ = self.temporal(feats)       # (B, T, hidden)
        out = out[:, -1, :]                 # last timestep

        logits = self.classifier(out)       # (B, n_classes)
        return logits


# For compatibility
Net = EEGSpectrogramMobileNetV2_LSTM