# unet_oasis_seg.py — Plain UNet for OASIS 2D segmentation (Task 1)
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
import torch.backends.cudnn as cudnn

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs): return x

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# =============================
# GLOBAL CONFIG
# =============================
# Set ROOT to your 'keras_png_slices_data' parent if not in the working dir.
ROOT: Union[str, Path, None] = None     # None -> ./keras_png_slices_data/keras_png_slices_data
EPOCHS: int       = 20
BATCH_SIZE: int   = 8
LR: float         = 2e-3
IMG_SIZE: int     = 256
USE_RGB: bool     = False               # MRI slices are usually single-channel
AUGMENT: bool     = True

# DataLoader
NUM_WORKERS: int         = 4
PERSISTENT_WORKERS: bool = False

# Segmentation setup
NUM_CLASSES: int = 2   # 2 for bg/fg; set to 4 if your masks are bg/CSF/GM/WM, etc.
IGNORE_INDEX: Optional[int] = None  # e.g., 255 if your masks contain "ignore"

# Early stop target (Task 1 target is ~0.90)
EARLY_STOP_DICE: float = 0.90

# Performance knobs
cudnn.benchmark = True
cudnn.deterministic = False

# -----------------------------
# Output folders
# -----------------------------
RUN_DIR   = Path("outputs") / f"UNet_OASIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIG_DIR   = RUN_DIR / "figs"
MODEL_DIR = RUN_DIR / "models"
for d in (FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =============================
# DATASET (images + seg masks)
# =============================
def _norm_key(name: str) -> str:
    # 'seg_441_slice_0.png' or 'case_441_slice_0.png' -> '441_slice_0.png'
    base = Path(name).name
    return base.split("_", 1)[1] if "_" in base else base

class SegSlicesDataset(Dataset):
    """
    Images: <root>/keras_png_slices_data/keras_png_slices_<split>/*.png
    Masks : <root>/keras_png_slices_data/keras_png_slices_seg_<split>/*.png
    Pairs by dropping the first token (seg_ vs case_).
    Masks can be palette-coded (e.g., 0/85/170/255) — we remap to 0..K-1 safely.
    """
    def __init__(
        self,
        split: str,                          # 'train'|'validate'|'test'
        root: Union[str, Path, None] = None,
        image_size: Tuple[int,int] = (256,256),
        use_rgb: bool = False,
        num_classes: int = 2,
        ignore_index: Optional[int] = None,
        augment: bool = False,
    ):
        if split not in {"train","validate","test"}:
            raise ValueError("split must be train|validate|test")

        try:
            here = Path(__file__).resolve().parent
        except NameError:
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

        self.img_paths: List[Path] = sorted(self.img_dir.glob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"No PNG images in {self.img_dir}")

        self.mask_index = { _norm_key(p.name): p for p in sorted(self.msk_dir.glob("*.png")) }
        missing = [p.name for p in self.img_paths if _norm_key(p.name) not in self.mask_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} masks missing in {self.msk_dir}. First few: {missing[:5]}")

        self.use_rgb = use_rgb
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.im_tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])
        self.msk_resize = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)

        self.augment = augment and split == "train"
        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.vflip = T.RandomVerticalFlip(p=0.5)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx: int):
        ip = self.img_paths[idx]
        key = _norm_key(ip.name)
        mp = self.mask_index[key]

        # image
        img = Image.open(ip)
        img = img.convert("RGB") if self.use_rgb else img.convert("L")
        x = self.im_tf(img)  # [C,H,W], in [0,1]

        # mask: load, resize, to numpy int
        m = Image.open(mp).convert("L")
        m = self.msk_resize(m)
        m = np.array(m, dtype=np.int64)

        # ---- remap mask intensities -> class indices [0..num_classes-1] ----
        vals = np.unique(m)

        # Exclude ignore index from mapping
        if self.ignore_index is not None:
            vals_no_ign = vals[vals != self.ignore_index]
        else:
            vals_no_ign = vals

        # If mask values exceed class index range (palette-coded), map sorted unique -> 0..K-1
        if vals_no_ign.size and vals_no_ign.max() > (self.num_classes - 1):
            sorted_vals = np.sort(vals_no_ign)
            lut = {v: i for i, v in enumerate(sorted_vals)}
            vmapped = np.full_like(m, fill_value=(self.ignore_index if self.ignore_index is not None else 0))
            if self.ignore_index is not None:
                valid = (m != self.ignore_index)
                vmapped[valid] = np.vectorize(lut.get)(m[valid])
            else:
                vmapped = np.vectorize(lut.get)(m)
            m = vmapped

        # Safety: clip any out-of-range labels
        if self.ignore_index is None:
            m = np.clip(m, 0, self.num_classes - 1)
        else:
            keep = (m != self.ignore_index)
            m[(keep) & (m >= self.num_classes)] = 0
            m[(keep) & (m < 0)] = 0

        y = torch.from_numpy(m)  # [H,W] long

        # paired spatial augmentations
        if self.augment:
            seed = np.random.randint(0, 10_000)
            torch.manual_seed(seed); x  = self.flip(x);  x  = self.vflip(x)
            torch.manual_seed(seed); yT = self.flip(y.unsqueeze(0).float()).squeeze(0).long()
            torch.manual_seed(seed); yT = self.vflip(yT.unsqueeze(0).float()).squeeze(0).long()
            y = yT

        return x, y, ip.name

# =============================
# UNet
# =============================
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
        # pad decoder to match skip spatially
        dy = x_skip.size(2) - x.size(2)
        dx = x_skip.size(3) - x.size(3)
        x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
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

# =============================
# Losses & metrics
# =============================
def soft_dice_per_class(
    logits, target, eps=1e-6, num_classes=None,
    ignore_index: Optional[int]=None, ignore_empty=True
):
    if num_classes is None:
        num_classes = logits.size(1)
    p = F.softmax(logits, dim=1)  # [B,C,H,W]

    if ignore_index is None:
        oh = F.one_hot(target, num_classes).permute(0,3,1,2).float()
        valid = torch.ones_like(target, dtype=torch.bool, device=target.device)
    else:
        valid = (target != ignore_index)
        t = torch.where(valid, target, torch.zeros_like(target))
        oh = F.one_hot(t, num_classes).permute(0,3,1,2).float()
        p  = p  * valid.unsqueeze(1)
        oh = oh * valid.unsqueeze(1)
        if not valid.any():
            return torch.zeros(num_classes, device=logits.device)

    dims = (0,2,3)
    inter = (p*oh).sum(dims)
    p_sum = p.sum(dims)
    gt_sum = oh.sum(dims)
    denom = p_sum + gt_sum

    dsc = (2*inter + eps) / (denom + eps)

    if ignore_empty:
        present = (gt_sum > 0)
        if present.any():
            return dsc[present]
        else:
            return torch.zeros_like(dsc)
    return dsc

def dice_plus_ce(logits, target, class_weights=None, ignore_index: Optional[int]=None):
    ce = F.cross_entropy(
        logits, target, weight=class_weights,
        ignore_index=(-100 if ignore_index is None else ignore_index)
    )
    dsc_present = soft_dice_per_class(
        logits, target, num_classes=logits.size(1),
        ignore_index=ignore_index, ignore_empty=True
    )
    dice_loss = 1 - (dsc_present.mean() if dsc_present.numel() > 0 else torch.tensor(0.0, device=logits.device))
    return ce + dice_loss, ce.detach(), dice_loss.detach(), dsc_present.detach()

@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index: Optional[int]=None):
    model.eval()
    dsc_list = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        dsc_present = soft_dice_per_class(
            logits, y, num_classes=num_classes,
            ignore_index=ignore_index, ignore_empty=True
        )
        if dsc_present.numel() > 0:
            dsc_list.append(dsc_present.cpu().numpy())
    if not dsc_list:
        return np.zeros(num_classes, dtype=np.float32)
    maxc = num_classes
    pad = []
    for arr in dsc_list:
        if arr.shape[0] == maxc:
            pad.append(arr)
        else:
            tmp = np.full((maxc,), np.nan, dtype=np.float32)
            tmp[:arr.shape[0]] = arr
            pad.append(tmp)
    per_class = np.nanmean(np.stack(pad, 0), axis=0)
    per_class = np.nan_to_num(per_class, nan=0.0)
    return per_class

# =============================
# Visualisation
# =============================
@torch.no_grad()
def save_overlays(model, loader, device, out_path, num_classes, n=6):
    model.eval()
    x, y, names = next(iter(loader))
    x = x[:n].to(device, non_blocking=True); y = y[:n]
    logits = model(x)
    pred = torch.argmax(logits, dim=1).cpu()

    palette = np.array([
        [0,0,0],[255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[255,0,255],[0,255,255]
    ], dtype=np.uint8)[:num_classes]

    rows=[]
    for i in range(min(n, x.size(0))):
        img = x[i].cpu(); img3 = img.repeat(3,1,1) if img.shape[0]==1 else img
        gt_idx = y[i].cpu().numpy()
        gt_idx = np.clip(gt_idx, 0, num_classes-1)
        gt_rgb = torch.from_numpy(palette[gt_idx]).permute(2,0,1)/255.0
        pr_rgb = torch.from_numpy(palette[pred[i].numpy()]).permute(2,0,1)/255.0
        rows.append(torch.cat([img3, gt_rgb, pr_rgb], dim=-1))
    grid = torch.cat(rows, dim=-2)
    save_image(grid, out_path)
    print(f"[saved] {out_path}")

def save_dice_bar(per_class, out_path, class_names=None):
    xs = np.arange(len(per_class))
    plt.figure(figsize=(max(5,len(per_class)*1.2),3.5))
    plt.bar(xs, per_class)
    plt.ylim(0,1); plt.ylabel("DSC (present classes avg)")
    if class_names and len(class_names)==len(per_class):
        plt.xticks(xs, class_names, rotation=45)
    else:
        plt.xticks(xs, [f"c{i}" for i in xs], rotation=45)
    plt.title("Per-class Dice (ignores absent classes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()
    print(f"[saved] {out_path}")

# =============================
# Main training loop
# =============================
def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    imsz = (IMG_SIZE, IMG_SIZE)

    # Datasets
    train_ds = SegSlicesDataset("train",    root=ROOT, image_size=imsz, use_rgb=USE_RGB,
                                num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, augment=AUGMENT)
    try:
        val_ds = SegSlicesDataset("validate", root=ROOT, image_size=imsz, use_rgb=USE_RGB,
                                  num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, augment=False)
    except FileNotFoundError:
        # Some OASIS mirrors don't have 'validate'—fallback to train as a sanity split
        val_ds = SegSlicesDataset("train",    root=ROOT, image_size=imsz, use_rgb=USE_RGB,
                                  num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, augment=False)
    test_ds = SegSlicesDataset("test",     root=ROOT, image_size=imsz, use_rgb=USE_RGB,
                               num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, augment=False)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

    # Model / opt
    model = UNet(in_channels=3 if USE_RGB else 1, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_mean = 0.0
    best_tag = "best_unet.pt"

    for ep in range(1, EPOCHS+1):
        model.train()
        tot=ce_s=dl_s=0.0; n=0

        pbar = tqdm(train_loader, desc=f"Epoch {ep:03d}/{EPOCHS} (train)", leave=False)
        for x,y,_ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss, ce, dl, _ = dice_plus_ce(logits, y, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            bs = x.size(0); n += bs
            tot += loss.item()*bs; ce_s += ce.item()*bs; dl_s += dl.item()*bs
            pbar.set_postfix({"loss": f"{tot/n:.3f}", "ce": f"{ce_s/n:.3f}",
                              "dice": f"{dl_s/n:.3f}"})

        per_class = evaluate(model, val_loader, device, NUM_CLASSES, ignore_index=IGNORE_INDEX)
        mean_dsc_present = float(np.mean(per_class[per_class>0]) if np.any(per_class>0) else 0.0)
        print(
            f"Epoch {ep:03d}/{EPOCHS} | loss={tot/n:.3f} "
            f"(ce={ce_s/n:.3f} dice={dl_s/n:.3f}) | "
            f"Val mean DSC(present)={mean_dsc_present:.4f} | " +
            " ".join([f"c{i}={per_class[i]:.3f}" for i in range(len(per_class))])
        )

        if ep % 5 == 0 or ep == 1:
            save_overlays(model, val_loader, device, FIG_DIR / f"val_overlay_e{ep:03d}.png", NUM_CLASSES, n=6)
            save_dice_bar(per_class, FIG_DIR / f"dice_val_e{ep:03d}.png")

        if mean_dsc_present > best_mean:
            best_mean = mean_dsc_present
            torch.save(model.state_dict(), MODEL_DIR / best_tag)

        # early stop at/above target Dice
        if mean_dsc_present >= EARLY_STOP_DICE:
            print(f"[early-stop] mean Dice (present classes) reached {mean_dsc_present:.4f} ≥ {EARLY_STOP_DICE}")
            break

    # Final evals & artifacts
    v_d = evaluate(model, val_loader, device, NUM_CLASSES, ignore_index=IGNORE_INDEX)
    t_d = evaluate(model, test_loader, device, NUM_CLASSES, ignore_index=IGNORE_INDEX)
    save_dice_bar(v_d, FIG_DIR / "dice_val_final.png")
    save_dice_bar(t_d, FIG_DIR / "dice_test_final.png")
    save_overlays(model, val_loader, device, FIG_DIR / "val_overlay_final.png", NUM_CLASSES, n=8)
    save_overlays(model, test_loader, device, FIG_DIR / "test_overlay_final.png", NUM_CLASSES, n=8)

    # Save final model (tag Dice & epochs in filename)
    tag = f"UNet_valDice{best_mean:.3f}_ep{ep:02d}.pt"
    torch.save(model.state_dict(), MODEL_DIR / tag)
    print(f"[VAL] per-class={np.round(v_d,4)} (mean over present={v_d[v_d>0].mean() if np.any(v_d>0) else 0.0:.4f})")
    print(f"[TEST] per-class={np.round(t_d,4)} (mean over present={t_d[t_d>0].mean() if np.any(t_d>0) else 0.0:.4f})")
    print(f"[saved] model -> {MODEL_DIR}")
    print(f"[saved] figs  -> {FIG_DIR}")

if __name__ == "__main__":
    main()
