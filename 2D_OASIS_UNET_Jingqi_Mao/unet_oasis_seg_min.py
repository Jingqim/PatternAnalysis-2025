# unet_oasis_seg_min.py — Minimal UNet for OASIS Task 1 (2D segmentation)
# NOTE: Always show the full script when changes are made.

from pathlib import Path
from datetime import datetime
from typing import Tuple, Union, List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Config (edit as needed)
# ----------------------------
ROOT: Union[str, Path, None] = None   # if None, expects ./keras_png_slices_data/keras_png_slices_data/...
IMG_SIZE: int   = 256
EPOCHS: int     = 50
BATCH_SIZE: int = 8
LR: float       = 2e-3

# Task-1 specifics
NUM_CLASSES: int = 4  # fixed 4 classes
EARLY_STOP_DICE: float = 0.90  # stop once mean Dice (present classes) reaches this

# Output dirs (timestamped)
RUN_DIR   = Path("outputs") / f"UNet_OASIS_MIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIG_DIR   = RUN_DIR / "figs"
MODEL_DIR = RUN_DIR / "models"
for d in (FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Dataset
# ----------------------------
def _norm_key(name: str) -> str:
    # 'seg_441_slice_0.png' or 'case_441_slice_0.png' -> '441_slice_0.png'
    base = Path(name).name
    return base.split("_", 1)[1] if "_" in base else base

class SegSlicesDataset(Dataset):
    """
    Minimal dataset for OASIS PNG slices.
    Requires split folders:
      images: keras_png_slices_<split>/*.png (e.g., case_*)
      masks : keras_png_slices_seg_<split>/*.png (e.g., seg_*)
    """
    def __init__(self, split: str, root: Union[str, Path, None], image_size: Tuple[int,int]=(256,256)):
        assert split in {"train","validate","test"}
        here = Path.cwd()
        outer_root = Path(root) if root is not None else here / "keras_png_slices_data"
        inner_root = outer_root / "keras_png_slices_data"

        split_map = {
            "train":    ("keras_png_slices_train",    "keras_png_slices_seg_train"),
            "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
            "test":     ("keras_png_slices_test",     "keras_png_slices_seg_test"),
        }
        img_dirname, msk_dirname = split_map[split]
        self.img_dir = inner_root / img_dirname
        self.msk_dir = inner_root / msk_dirname

        if not self.img_dir.exists() or not self.msk_dir.exists():
            raise FileNotFoundError(f"Expected split folders missing: {self.img_dir} and/or {self.msk_dir}")

        self.img_paths: List[Path] = sorted(self.img_dir.glob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"No PNG images in {self.img_dir}")

        self.mask_index = { _norm_key(p.name): p for p in sorted(self.msk_dir.glob("*.png")) }
        missing = [p.name for p in self.img_paths if _norm_key(p.name) not in self.mask_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} masks missing in {self.msk_dir}. Example: {missing[:5]}")

        self.im_tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0,1], shape [1,H,W]
        ])
        self.msk_resize = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx: int):
        ip = self.img_paths[idx]
        mp = self.mask_index[_norm_key(ip.name)]

        # image (single-channel MRI)
        x = Image.open(ip).convert("L")
        x = self.im_tf(x)  # [1,H,W]

        # mask (grayscale palette → class indices 0..3)
        m = Image.open(mp).convert("L")
        m = self.msk_resize(m)
        m = np.array(m, dtype=np.int64)

        # If masks are palette-coded like {0,85,170,255}, map sorted unique -> {0,1,2,3}
        vals = np.unique(m)
        if vals.max() > (NUM_CLASSES - 1):
            sorted_vals = np.sort(vals)
            lut = {v: i for i, v in enumerate(sorted_vals)}
            m = np.vectorize(lut.get)(m).astype(np.int64)

        # Clamp to [0..3]
        m = np.clip(m, 0, NUM_CLASSES-1)
        y = torch.from_numpy(m).long()  # [H,W], int64 indices

        return x, y, ip.name


# ----------------------------
# Model (UNet, minimal)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x_dec, x_skip):
        x = self.up(x_dec)
        # pad to match
        dy = x_skip.size(2) - x.size(2)
        dx = x_skip.size(3) - x.size(3)
        x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1   = Up(512, 512, 256)
        self.up2   = Up(256, 256, 128)
        self.up3   = Up(128, 128, 64)
        self.up4   = Up(64,  64,  64)
        self.outc  = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.down4(x4)
        y = self.up1(xb, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        return self.outc(y)


# ----------------------------
# Loss / Metric / Viz (minimal)
# ----------------------------
def soft_dice_per_class(logits, target, eps=1e-6, num_classes=4):
    p = F.softmax(logits, dim=1)                      # [B,C,H,W]
    oh = F.one_hot(target, num_classes).permute(0,3,1,2).float()  # [B,C,H,W]
    dims = (0,2,3)
    inter = (p*oh).sum(dims)
    p_sum = p.sum(dims)
    gt_sum = oh.sum(dims)
    dsc = (2*inter + eps) / (p_sum + gt_sum + eps)    # per-class
    # average only over classes that appear in GT to avoid penalizing absent ones
    present = (gt_sum > 0)
    if present.any():
        return dsc[present]
    else:
        return torch.zeros(1, device=logits.device)

def dice_plus_ce(logits, target):
    ce = F.cross_entropy(logits, target)
    dsc_present = soft_dice_per_class(logits, target, num_classes=NUM_CLASSES)
    dice_loss = 1 - (dsc_present.mean() if dsc_present.numel() > 0 else torch.tensor(0.0, device=logits.device))
    return ce + dice_loss, ce.detach(), dice_loss.detach(), dsc_present.detach()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    per_batch = []
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        dsc_present = soft_dice_per_class(logits, y, num_classes=NUM_CLASSES)
        if dsc_present.numel() > 0:
            per_batch.append(dsc_present.mean().item())
    return float(np.mean(per_batch)) if per_batch else 0.0

@torch.no_grad()
def save_overlays(model, loader, device, out_path, n=6):
    model.eval()
    x, y, _ = next(iter(loader))
    x = x[:n].to(device); y = y[:n]
    logits = model(x)
    pred = torch.argmax(logits, dim=1).cpu()

    # fixed 4-class grayscale palette
    palette = np.array([
        [0, 0, 0],         # class 0: black
        [85, 85, 85],      # class 1: dark gray
        [170, 170, 170],   # class 2: light gray
        [255, 255, 255],   # class 3: white
    ], dtype=np.uint8)

    rows=[]
    for i in range(min(n, x.size(0))):
        img = x[i].float().cpu()
        img3 = img.repeat(3,1,1)  # grayscale to 3ch for concat
        gt_idx = torch.clamp(y[i].cpu(), 0, NUM_CLASSES-1).numpy()
        pr_idx = pred[i].numpy()
        gt_rgb = torch.from_numpy(palette[gt_idx]).permute(2,0,1)/255.0
        pr_rgb = torch.from_numpy(palette[pr_idx]).permute(2,0,1)/255.0
        rows.append(torch.cat([img3, gt_rgb, pr_rgb], dim=-1))
    grid = torch.cat(rows, dim=-2)
    save_image(grid, out_path)

def save_dice_bar(scores: List[float], out_path: Path, title: str):
    plt.figure(figsize=(5,3.5))
    plt.bar([0], [scores[0] if scores else 0.0], color="gray")
    plt.ylim(0,1)
    plt.xticks([0], ["mean Dice (present)"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()


# ----------------------------
# Training loop (minimal)
# ----------------------------
def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    imsz = (IMG_SIZE, IMG_SIZE)

    # Strict: require all three splits to exist (no hidden fallbacks)
    train_ds = SegSlicesDataset("train",    root=ROOT, image_size=imsz)
    val_ds   = SegSlicesDataset("validate", root=ROOT, image_size=imsz)
    test_ds  = SegSlicesDataset("test",     root=ROOT, image_size=imsz)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = 0.0
    best_tag = "best_unet.pt"

    for ep in range(1, EPOCHS+1):
        model.train()
        total=ce_s=dl_s=cnt=0.0
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss, ce, dl, _ = dice_plus_ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = x.size(0)
            total += loss.item()*bs; ce_s += ce.item()*bs; dl_s += dl.item()*bs; cnt += bs

        val_mean_dice = evaluate(model, val_loader, device)
        print(f"Epoch {ep:03d}/{EPOCHS} | loss={total/cnt:.3f} (ce={ce_s/cnt:.3f}, dice={dl_s/cnt:.3f}) | "
              f"Val mean Dice(present)={val_mean_dice:.4f}")

        if val_mean_dice > best_val:
            best_val = val_mean_dice
            torch.save(model.state_dict(), MODEL_DIR / best_tag)

        if val_mean_dice >= EARLY_STOP_DICE:
            print(f"[early-stop] mean Dice reached {val_mean_dice:.4f} ≥ {EARLY_STOP_DICE}")
            break

    # Final evaluation & artifacts
    val_final = evaluate(model, val_loader, device)
    test_final = evaluate(model, test_loader, device)
    save_dice_bar([val_final], FIG_DIR / "dice_val_final.png",  "Validation mean Dice (present)")
    save_dice_bar([test_final], FIG_DIR / "dice_test_final.png", "Test mean Dice (present)")
    save_overlays(model, val_loader,  device, FIG_DIR / "val_overlay_final.png",  n=8)
    save_overlays(model, test_loader, device, FIG_DIR / "test_overlay_final.png", n=8)

    # Save final model (tag)
    tag = f"UNet_valDice{best_val:.3f}_ep{ep:02d}.pt"
    torch.save(model.state_dict(), MODEL_DIR / tag)
    print(f"[VAL]  mean Dice(present)={val_final:.4f}")
    print(f"[TEST] mean Dice(present)={test_final:.4f}")
    print(f"[saved] models -> {MODEL_DIR}")
    print(f"[saved] figs   -> {FIG_DIR}")

if __name__ == "__main__":
    main()
