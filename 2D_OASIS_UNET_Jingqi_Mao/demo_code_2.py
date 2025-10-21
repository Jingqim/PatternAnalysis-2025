# oasis_unet_train.py
# Single-file minimal trainer for OASIS 2D PNG multi-class segmentation (4 classes)
# Uses SIX explicit folders:
#   - Real images:      IMG_TRAIN, IMG_VALIDATE, IMG_TEST
#   - Segmentation PNGs:MASK_TRAIN, MASK_VALIDATE, MASK_TEST
# Requirements:
#   * Each dir contains PNGs with matching basenames (e.g., 0001.png in both image & mask dirs).
#   * Images: grayscale 256x256 (will be resized if different)
#   * Masks:  integer labels {0,1,2,3} as grayscale PNG values (clipped to [0,3])
#
# Functionality: dataset -> model -> train/eval -> mean Dice -> early stop -> save best -> loss plot

import os, time
from pathlib import Path
from typing import Tuple, Dict, Sequence
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# EDIT THESE SIX PATHS
# -----------------------
# \keras_png_slices_data\keras_png_slices_seg_test
IMG_TRAIN     = r"./keras_png_slices_data/keras_png_slices_train"
IMG_VALIDATE  = r"./keras_png_slices_data/keras_png_slices_validate"
IMG_TEST      = r"./keras_png_slices_data/keras_png_slices_test"

MASK_TRAIN    = r"./keras_png_slices_data/keras_png_slices_seg_train"
MASK_VALIDATE = r"./keras_png_slices_data/keras_png_slices_seg_validate"
MASK_TEST     = r"./keras_png_slices_data/keras_png_slices_seg_test"


# -----------------------
# HYPERPARAMS (simple)
# -----------------------
NUM_CLASSES      = 4
BATCH_SIZE       = 8
LR               = 5e-3
EPOCHS           = 3
EARLY_STOP_DICE  = 0.90
NUM_WORKERS      = 2
RESIZE           = (256, 256)     # keep native size
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR         = Path("oasis_unet_model_using_classdemo"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Utility
# -----------------------
def _basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _collect_pngs(d: str) -> Sequence[str]:
    return sorted([f for f in os.listdir(d) if f.lower().endswith(".png")])

# -----------------------
# Dataset
# -----------------------
class OASISPNGSegmentation(Dataset):
    """
    Minimal PNG dataset for paired image/mask folders.
    - img_dir:  directory with grayscale image PNGs
    - mask_dir: directory with label PNGs (0..3)
    Filenames must match by basename (e.g., 001.png in both).
    """
    def __init__(self, img_dir: str, mask_dir: str, resize: Tuple[int,int]=(256,256)):
        assert os.path.isdir(img_dir),  f"Missing {img_dir}"
        assert os.path.isdir(mask_dir), f"Missing {mask_dir}"
        self.img_dir  = img_dir
        self.mask_dir = mask_dir

        img_files  = _collect_pngs(img_dir)
        mask_files = _collect_pngs(mask_dir)

        img_set  = {_basename_noext(f) for f in img_files}
        mask_set = {_basename_noext(f) for f in mask_files}
        common   = sorted(list(img_set & mask_set))
        assert len(common) > 0, f"No matching image/mask pairs between {img_dir} and {mask_dir}."

        self.samples = [(os.path.join(self.img_dir,  b + ".png"),
                         os.path.join(self.mask_dir, b + ".png")) for b in common]

        h, w = resize
        self.img_tf = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                     # -> [1,H,W] in [0,1]
        ])
        self.mask_resize = T.Resize((h, w), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Image: grayscale float tensor [1,H,W]
        img = Image.open(img_path).convert("L")
        img = self.img_tf(img)

        # Mask: integer class ids [H,W] in {0,1,2,3}
        m = Image.open(mask_path).convert("L")
        m = self.mask_resize(m)
        m = np.array(m, dtype=np.uint8)
        m = np.clip(m, 0, NUM_CLASSES-1)
        mask = torch.from_numpy(m).long()      # [H,W], Long

        return img, mask

def make_loaders(
    img_train: str, img_val: str, img_test: str,
    mask_train: str, mask_val: str, mask_test: str,
    batch_size=8, num_workers=2, resize=(256,256)
):
    train_ds = OASISPNGSegmentation(img_train,  mask_train,  resize)
    val_ds   = OASISPNGSegmentation(img_val,    mask_val,    resize)
    test_ds  = OASISPNGSegmentation(img_test,   mask_test,   resize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# -----------------------
# Model (small U-Net)
# -----------------------
def conv_block(cin, cout, drop=0.0):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(drop),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(drop),
    )

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_ch=32, drop=0.0):
        super().__init__()
        self.enc1 = conv_block(in_channels, base_ch, drop)
        self.enc2 = conv_block(base_ch, base_ch*2, drop)
        self.enc3 = conv_block(base_ch*2, base_ch*4, drop)

        self.pool = nn.MaxPool2d(2)

        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = conv_block(base_ch*4 + base_ch*2, base_ch*2, drop)

        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = conv_block(base_ch*2 + base_ch, base_ch, drop)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)               # [B,C,256,256]
        e2 = self.enc2(self.pool(e1))   # [B,2C,128,128]
        e3 = self.enc3(self.pool(e2))   # [B,4C, 64, 64]

        d2 = self.up2(e3)               # [B,4C,128,128]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)              # [B,2C,128,128]

        d1 = self.up1(d2)               # [B,2C,256,256]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)              # [B,C,256,256]

        logits = self.out_conv(d1)      # [B,num_classes,256,256] (raw logits)
        return logits

# -----------------------
# Metrics / plotting
# -----------------------
def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    pred:   [B,H,W] argmax predictions (Long)
    target: [B,H,W] ground truth (Long)
    returns [num_classes] dice per class
    """
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum(dim=(1,2))
        denom = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
        sample_dice = torch.where(denom > 0, (2*inter) / denom.clamp(min=1e-6), torch.ones_like(denom))
        dices.append(sample_dice.mean())
    return torch.stack(dices)

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_dc = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)             # [B,1,H,W]
            masks = masks.to(device)            # [B,H,W]
            logits = model(imgs)                # [B,C,H,W]
            loss = ce(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1) # [B,H,W]
            all_dc.append(dice_per_class(preds, masks, NUM_CLASSES).unsqueeze(0))
    avg_loss  = total_loss / len(loader.dataset)
    mean_dice = torch.cat(all_dc, dim=0).mean(dim=0).mean().item()
    return {"loss": avg_loss, "mean_dice": float(mean_dice)}

def plot_curve(train_hist, val_hist, out_png):
    plt.figure()
    plt.plot([h["loss"] for h in train_hist], label="train_loss")
    plt.plot([h["loss"] for h in val_hist],   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("CE loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

# -----------------------
# Train
# -----------------------
def main():
    train_loader, val_loader, test_loader = make_loaders(
        IMG_TRAIN, IMG_VALIDATE, IMG_TEST,
        MASK_TRAIN, MASK_VALIDATE, MASK_TEST,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, resize=RESIZE
    )

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    ce    = nn.CrossEntropyLoss()

    best_val = -1.0
    train_hist, val_hist = [], []

    print(f"Device: {DEVICE} | epochs={EPOCHS} | batch_size={BATCH_SIZE}")

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs)
            loss   = ce(logits, masks)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item() * imgs.size(0)

        train_loss = running / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, DEVICE)
        train_hist.append({"loss": train_loss})
        val_hist.append({"loss": val_metrics["loss"], "mean_dice": val_metrics["mean_dice"]})

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_meanDice={val_metrics['mean_dice']:.4f}")

        # Save best
        if val_metrics["mean_dice"] > best_val:
            best_val = val_metrics["mean_dice"]
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = SAVE_DIR / f"{ts}_dice{best_val:.2f}_ep{epoch}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_mean_dice": best_val
            }, out_path)
            print(f"Saved: {out_path}")

        # Early stop
        if best_val >= EARLY_STOP_DICE:
            print(f"Early stopping at epoch {epoch}: val mean Dice {best_val:.3f} ≥ {EARLY_STOP_DICE}.")
            break

    plot_curve(train_hist, val_hist, SAVE_DIR / "loss_curve.png")

    test_metrics = evaluate(model, test_loader, DEVICE)
    print(f"TEST: loss={test_metrics['loss']:.4f} | meanDice={test_metrics['mean_dice']:.4f}")

if __name__ == "__main__":
    main()
