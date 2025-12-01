import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


class PerceptualNetwork(nn.Module):
    def __init__(
            self,
            arch="vgg16",
            layers={'3': "relu1_2", '8': "relu2_2", '15': "relu3_3", '22': "relu4_3"}
    ):
        super().__init__()
        assert hasattr(models, arch)
        # Use new weights API to avoid deprecation warnings
        if arch == "vgg16":
            weights = VGG16_Weights.IMAGENET1K_V1
            self.net = getattr(models, arch)(weights=weights).features
        else:
            # For other architectures, try to use weights enum if available
            try:
                weights_enum = getattr(models, f"{arch.upper()}_Weights", None)
                if weights_enum is not None:
                    self.net = getattr(models, arch)(weights=weights_enum.DEFAULT).features
                else:
                    # Fallback to old API
                    self.net = getattr(models, arch)(pretrained=True).features
            except (AttributeError, TypeError):
                # Fallback to old API for compatibility
                self.net = getattr(models, arch)(pretrained=True).features
        self.layers = layers

    def forward(self, x):
        out = {}
        for name, m in self.net._modules.items():
            x = m(x)
            if name in self.layers:
                out[name] = x
        return out


class PerceptualLoss(nn.Module):
    def __init__(
            self,
            size=None,
            arch="vgg16",
            layers={'3': "relu1_2", '8': "relu2_2", '15': "relu3_3", '22': "relu4_3"}
    ):
        super().__init__()
        self.size = size
        self.net = PerceptualNetwork(arch, layers)

    def forward(self, pred, gt):
        if self.size is not None:
            pred = self._resize(pred)
            gt = self._resize(gt).detach()
        pred_out = self.net(pred)
        with torch.no_grad():
            pred_gt = self.net(gt)
        loss = 0
        for k, v in pred_out.items():
            loss += F.l1_loss(v, pred_gt[k])
        return loss

    def _resize(self, img):
        return F.interpolate(
            img,
            size=self.size,
            mode="bilinear",
            align_corners=False
        )
