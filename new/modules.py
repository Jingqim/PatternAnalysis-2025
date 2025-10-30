# modules.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseResNet18(nn.Module):
    """Siamese encoder built from ResNet18 trained from scratch.
    Produces L2-normalized 512-d embeddings and cosine logits."""
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None)
        m.fc = nn.Identity()  
        self.encoder = m
        self.scale = nn.Parameter(torch.tensor(10.0))  

    def forward_once(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=1)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        logits = self.scale * F.cosine_similarity(z1, z2)  # (B,)
        return logits, z1, z2


class ModelEMA:
    """Exponential Moving Average for evaluation and hard-negative mining."""
    def __init__(self, model: nn.Module, decay=0.995, device=None):
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(self.decay * v + (1.0 - self.decay) * msd[k].detach())
