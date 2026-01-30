# xai_gradcam.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer="features.6"):
        """
        model: your CNN model
        target_layer: name of last convolution layer to hook (e.g., 'features.6')
        """
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        layer = dict(model.named_modules())[target_layer]
        layer.register_forward_hook(self._hook_fwd)
        layer.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, m, i, o):
        self.activations = o

    def _hook_bwd(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x, class_idx=1):
        """
        x: torch tensor of shape [1, C, H, W]
        class_idx: which class to visualize (1 = seizure)
        """
        self.model.zero_grad()
        out = self.model(x)
        out[:, class_idx].sum().backward()

        A = self.activations        # [1, K, h, w]
        G = self.gradients          # [1, K, h, w]
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap(spectrogram_2d, heatmap_2d, alpha=0.45):
    """
    spectrogram_2d: numpy array (H, W)
    heatmap_2d: numpy array (H, W) with values in [0,1]
    Returns: PIL image with overlay
    """
    plt.figure()
    plt.imshow(spectrogram_2d, cmap="magma", aspect="auto", origin="lower")
    plt.imshow(heatmap_2d, cmap="jet", alpha=alpha, aspect="auto", origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("gradcam_overlay_tmp.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return Image.open("gradcam_overlay_tmp.png")
