# train.py — UNet trainer with tqdm, cuDNN autotune, early-stop at Dice ≥ 0.90
# Uses ALL training data by default; set train_limit to cap samples if desired.

import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset2 import create_dataloaders_here

# ---- cuDNN autotune for faster convs on fixed sizes ----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# -------------------
# Simple U-Net
# -------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base=32):
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base*2)
        self.e3 = conv_block(base*2, base*4)
        self.e4 = conv_block(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(base*8, base*16)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.d4 = conv_block(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.d3 = conv_block(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.d2 = conv_block(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.d1 = conv_block(base*2, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b  = self.bott(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)  # logits [B,C,H,W]

# -------------------
# Loss & metric
# -------------------
class DiceCE(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.n_classes = n_classes
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.n_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2*inter + 1e-6) / (denom + 1e-6)
        return ce + (1 - dice.mean())

@torch.no_grad()
def eval_mean_dice(model: nn.Module, loader: DataLoader, n_classes: int, device: str) -> float:
    model.eval()
    dices = []
    for x, y in tqdm(loader, desc="Validating", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = F.softmax(model(x), dim=1)
        oh = F.one_hot(y, n_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (p*oh).sum(dims)
        denom = p.sum(dims) + oh.sum(dims)
        d = (2*inter + 1e-6)/(denom + 1e-6)
        dices.append(d)
    if not dices:
        return 0.0
    per_class = torch.cat(dices,0).mean(0)
    return float(per_class.mean())

# -------------------
# Trainer
# -------------------
def train_unet(
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    base_channels: int = 32,
    target_dice: float = 0.90,
    size_hw=(256,256),
    train_limit=None,          # None => use ALL training samples
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build loaders from dataset.py
    full_train_ld, val_ld, _ = create_dataloaders_here(
        batch_size=batch_size, size_hw=size_hw, augment_train=True
    )
    full_train_ds = full_train_ld.dataset
    total = len(full_train_ds)

    # Coerce/guard train_limit
    if train_limit is not None:
        try:
            train_limit = int(train_limit)
        except (TypeError, ValueError):
            train_limit = None

    if train_limit is None or train_limit <= 0 or train_limit >= total:
        train_ld = full_train_ld
        print(f"Training on ALL {total} samples | Val: {len(val_ld.dataset)}")
    else:
        idx = range(train_limit)
        train_ds = Subset(full_train_ds, idx)
        # Use robust loader settings for throughput
        nworkers = max(2, (os.cpu_count() or 4) // 2)
        train_ld = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=nworkers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )
        print(f"Training on {train_limit} / {total} samples | Val: {len(val_ld.dataset)}")

    # Infer n_classes from a small batch
    xb, yb = next(iter(train_ld))
    n_classes = int(yb.max().item()) + 1
    print(f"Detected n_classes={n_classes} | Device={device}")

    model = UNet2D(in_ch=1, n_classes=n_classes, base=base_channels).to(device)
    loss_fn = DiceCE(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = -1.0
    dice_hist = []

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(train_ld, desc=f"Epoch {ep}/{epochs}", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running / max(1, len(train_ld))
        val_dice = eval_mean_dice(model, val_ld, n_classes, device)
        dice_hist.append(val_dice)
        tqdm.write(f"Epoch {ep:02d}: train loss {avg_loss:.4f} | val Dice {val_dice:.4f}")

        if val_dice >= target_dice:
            tqdm.write(f"Early stop: Dice {val_dice:.4f} ≥ {target_dice}")
            _save_artifacts(model, dice_hist, val_dice, ep)
            return

        if val_dice > best_val:
            best_val = val_dice

    final_dice = dice_hist[-1] if dice_hist else 0.0
    _save_artifacts(model, dice_hist, final_dice, epochs)

def _save_artifacts(model: nn.Module, dice_hist, val_dice: float, epochs_done: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"oasis_unet_model_{ts}_dice{val_dice:.3f}_ep{epochs_done:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")

    plt.figure()
    plt.plot(range(1, len(dice_hist)+1), dice_hist)
    plt.xlabel("Epoch"); plt.ylabel("Validation Mean Dice")
    plt.title("Validation Dice over Epochs"); plt.grid(True)
    plt.savefig(out_dir / "dice_curve.png", bbox_inches="tight")
    plt.close()

    print(f"Saved model: {out_dir/'model.pt'}")
    print(f"Saved dice figure: {out_dir/'dice_curve.png'}")

if __name__ == "__main__":
    train_unet(
        epochs=10,
        batch_size=8,
        lr=1e-3,
        base_channels=32,
        target_dice=0.90,
        size_hw=(256,256),
        train_limit=None,   # set e.g. 100 or 10000 to limit; None uses ALL data
    )

