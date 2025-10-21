# unet_oasis_seg_win2080.py — UNet for OASIS 2D segmentation (Windows + RTX 2080 tuned)
# Features:
# - AMP (FP16) + channels_last + cudnn.benchmark for speed
# - Safe torch.compile(): auto-disabled on Windows / no Triton
# - Robust mask remap (palette→indices), IGNORE_INDEX support
# - CLI flags (batch/epochs/workers/etc.)
# - Optional gradient accumulation
# - Simple throughput meter (images/sec)

from pathlib import Path
from datetime import datetime
from typing import Tuple, Union, List, Optional
import os, platform, argparse, time

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def _has_triton():
    return False

def _norm_key(name: str) -> str:
    base = Path(name).name
    return base.split("_", 1)[1] if "_" in base else base


# ----------------------------
# Dataset
# ----------------------------
class SegSlicesDataset(Dataset):
    """
    Images: <root>/keras_png_slices_data/keras_png_slices_<split>/*.png  e.g. case_***
    Masks : <root>/keras_png_slices_data/keras_png_slices_seg_<split>/*.png  e.g. seg_***
    Pairs by dropping the first token, robustly remaps palette intensities -> class indices.
    """
    def __init__(
        self,
        split: str,                          # 'train'|'validate'|'test'
        root: Union[str, Path, None],
        image_size: Tuple[int,int],
        use_rgb: bool,
        num_classes: int,
        ignore_index: Optional[int],
        augment: bool,
    ):
        if split not in {"train","validate","test"}:
            raise ValueError("split must be train|validate|test")

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

        # mask
        m = Image.open(mp).convert("L")
        m = self.msk_resize(m)
        m = np.array(m, dtype=np.int64)

        # remap palette to indices
        vals = np.unique(m)
        vals_no_ign = vals[vals != self.ignore_index] if (self.ignore_index is not None) else vals
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

        if self.ignore_index is None:
            m = np.clip(m, 0, self.num_classes - 1)
        else:
            keep = (m != self.ignore_index)
            m[(keep) & (m >= self.num_classes)] = 0
            m[(keep) & (m < 0)] = 0

        y = torch.from_numpy(m)  # [H,W] long

        if self.augment:
            seed = np.random.randint(0, 10_000)
            torch.manual_seed(seed); x  = self.flip(x);  x  = self.vflip(x)
            torch.manual_seed(seed); yT = self.flip(y.unsqueeze(0).float()).squeeze(0).long()
            torch.manual_seed(seed); yT = self.vflip(yT.unsqueeze(0).float()).squeeze(0).long()
            y = yT

        return x, y, ip.name


# ----------------------------
# Model
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


# ----------------------------
# Losses / Metrics / Viz
# ----------------------------
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
        return dsc[present] if present.any() else torch.zeros_like(dsc)
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
def evaluate(model, loader, device, num_classes, ignore_index: Optional[int], amp_enabled: bool, amp_dtype):
    model.eval()
    dsc_list = []
    use_amp = (amp_enabled and device.type == "cuda")
    if use_amp:
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        ctx = _NullCtx()

    with ctx:
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            dsc_present = soft_dice_per_class(
                logits, y, num_classes=num_classes,
                ignore_index=ignore_index, ignore_empty=True
            )
            if dsc_present.numel() > 0:
                dsc_list.append(dsc_present.float().cpu().numpy())

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

@torch.no_grad()
def save_overlays(model, loader, device, out_path, num_classes, n=6, amp_enabled=True, amp_dtype=torch.float16):
    model.eval()
    x, y, names = next(iter(loader))
    x = x[:n].to(device, non_blocking=True)
    y = y[:n]

    use_amp = (amp_enabled and device.type == "cuda")
    if use_amp:
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        ctx = _NullCtx()

    with ctx:
        logits = model(x)
    pred = torch.argmax(logits, dim=1).cpu()

    palette = np.array([
        [0,0,0],[255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[255,0,255],[0,255,255]
    ], dtype=np.uint8)[:num_classes]

    rows=[]
    for i in range(min(n, x.size(0))):
        img = x[i].float().cpu()
        img3 = img.repeat(3,1,1) if img.shape[0]==1 else img
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


# ----------------------------
# Training
# ----------------------------
def main(args):
    # --- Perf knobs ---
    cudnn.benchmark   = True
    cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False  # Turing has no TF32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Decide if torch.compile is available (Windows usually lacks Triton)
    CAN_COMPILE = (args.use_compile
                   and device.type == "cuda"
                   and _has_triton()
                   and platform.system().lower() != "windows")
    if args.use_compile and not CAN_COMPILE:
        reasons = []
        if device.type != "cuda": reasons.append("no CUDA")
        if platform.system().lower() == "windows": reasons.append("Windows OS")
        if not _has_triton(): reasons.append("Triton not installed")
        print(f"[compile] disabled ({', '.join(reasons)}) — using eager mode")

    # Datasets
    imsz = (args.img_size, args.img_size)
    train_ds = SegSlicesDataset("train",    root=args.root, image_size=imsz, use_rgb=args.use_rgb,
                                num_classes=args.num_classes, ignore_index=args.ignore_index, augment=not args.no_augment)
    try:
        val_ds = SegSlicesDataset("validate", root=args.root, image_size=imsz, use_rgb=args.use_rgb,
                                  num_classes=args.num_classes, ignore_index=args.ignore_index, augment=False)
    except FileNotFoundError:
        val_ds = SegSlicesDataset("train",    root=args.root, image_size=imsz, use_rgb=args.use_rgb,
                                  num_classes=args.num_classes, ignore_index=args.ignore_index, augment=False)
    test_ds = SegSlicesDataset("test",     root=args.root, image_size=imsz, use_rgb=args.use_rgb,
                               num_classes=args.num_classes, ignore_index=args.ignore_index, augment=False)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent, prefetch_factor=args.prefetch
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent, prefetch_factor=args.prefetch
    )
    test_loader  = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent, prefetch_factor=args.prefetch
    )

    # Outputs
    run_dir   = Path(args.out_dir) / f"UNet_OASIS_win2080_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fig_dir   = run_dir / "figs"
    model_dir = run_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = UNet(in_channels=3 if args.use_rgb else 1, num_classes=args.num_classes)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # torch.compile
    if CAN_COMPILE:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[compile] model compiled with mode='{args.compile_mode}'")
        except Exception as e:
            print(f"[compile] skipped at runtime, falling back (reason: {e})")
            CAN_COMPILE = False

    # Optimizer + AMP scaler
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    from torch import amp as _amp
    scaler = _amp.GradScaler("cuda", enabled=(args.amp and device.type=="cuda"))

    best_mean = 0.0
    best_tag = "best_unet.pt"

    # Throughput meter (images/sec, per epoch)
    def _throughput(num_imgs, t0, t1): 
        dt = max(t1 - t0, 1e-6); return num_imgs / dt

    for ep in range(1, args.epochs+1):
        model.train()
        tot=ce_s=dl_s=0.0; n=0
        seen_images = 0
        t_epoch0 = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {ep:03d}/{args.epochs} (train)", leave=False)
        opt.zero_grad(set_to_none=True)
        for step, (x,y,_) in enumerate(pbar, start=1):
            # Move to GPU
            if device.type == "cuda" and args.channels_last:
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # AMP forward
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type=="cuda")):
                logits = model(x)
                loss, ce, dl, _ = dice_plus_ce(logits, y, ignore_index=args.ignore_index)
                if args.accum_steps > 1:
                    loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            # Step on accumulation boundary
            if step % args.accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            bs = x.size(0); n += bs; seen_images += bs
            tot += float(loss)*bs if args.accum_steps == 1 else float(loss)*bs*args.accum_steps
            ce_s += float(ce)*bs; dl_s += float(dl)*bs
            pbar.set_postfix({"loss": f"{tot/n:.3f}", "ce": f"{ce_s/n:.3f}", "dice": f"{dl_s/n:.3f}"})

        # If steps didn't align with accum, do a final optimizer step
        if (len(train_loader) % args.accum_steps) != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        t_epoch1 = time.perf_counter()
        imgs_per_sec = _throughput(seen_images, t_epoch0, t_epoch1)

        # Validation
        per_class = evaluate(model, val_loader, device, args.num_classes, args.ignore_index, args.amp, torch.float16)
        mean_dsc_present = float(np.mean(per_class[per_class>0]) if np.any(per_class>0) else 0.0)
        print(
            f"Epoch {ep:03d}/{args.epochs} | loss={tot/n:.3f} "
            f"(ce={ce_s/n:.3f} dice={dl_s/n:.3f}) | "
            f"Val mean DSC(present)={mean_dsc_present:.4f} | "
            f"imgs/s={imgs_per_sec:.1f} | " +
            " ".join([f"c{i}={per_class[i]:.3f}" for i in range(len(per_class))])
        )

        if (ep % args.save_every == 0) or (ep == 1):
            save_overlays(model, val_loader, device, fig_dir / f"val_overlay_e{ep:03d}.png",
                          args.num_classes, n=6, amp_enabled=args.amp, amp_dtype=torch.float16)
            save_dice_bar(per_class, fig_dir / f"dice_val_e{ep:03d}.png")

        if mean_dsc_present > best_mean:
            best_mean = mean_dsc_present
            torch.save(model.state_dict(), model_dir / best_tag)

        if mean_dsc_present >= args.early_stop_dice:
            print(f"[early-stop] mean Dice (present classes) reached {mean_dsc_present:.4f} ≥ {args.early_stop_dice}")
            break

    # Final evals & artifacts
    v_d = evaluate(model, val_loader, device, args.num_classes, args.ignore_index, args.amp, torch.float16)
    t_d = evaluate(model, test_loader, device, args.num_classes, args.ignore_index, args.amp, torch.float16)
    save_dice_bar(v_d, fig_dir / "dice_val_final.png")
    save_dice_bar(t_d, fig_dir / "dice_test_final.png")
    save_overlays(model, val_loader, device, fig_dir / "val_overlay_final.png", args.num_classes, n=8,
                  amp_enabled=args.amp, amp_dtype=torch.float16)
    save_overlays(model, test_loader, device, fig_dir / "test_overlay_final.png", args.num_classes, n=8,
                  amp_enabled=args.amp, amp_dtype=torch.float16)

    tag = f"UNet_valDice{best_mean:.3f}_ep{ep:02d}.pt"
    torch.save(model.state_dict(), model_dir / tag)
    print(f"[VAL]  per-class={np.round(v_d,4)} (mean over present={v_d[v_d>0].mean() if np.any(v_d>0) else 0.0:.4f})")
    print(f"[TEST] per-class={np.round(t_d,4)} (mean over present={t_d[t_d>0].mean() if np.any(t_d>0) else 0.0:.4f})")
    print(f"[saved] model -> {model_dir}")
    print(f"[saved] figs  -> {fig_dir}")


# ----------------------------
# CLI
# ----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="UNet OASIS (Windows + RTX2080 optimised)")
    # data
    p.add_argument("--root", type=str, default=None, help="Folder that contains keras_png_slices_data/")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--use-rgb", action="store_true", help="Use RGB inputs (default: grayscale)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--ignore-index", type=int, default=None)
    p.add_argument("--no-augment", action="store_true")

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--early-stop-dice", type=float, default=0.90)
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")

    # perf toggles
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP mixed precision")
    p.add_argument("--channels-last", dest="channels_last", action="store_true", help="Use NHWC memory format")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--no-persistent", action="store_true")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))

    # compile
    p.add_argument("--use-compile", action="store_true", help="Try torch.compile if Triton/Linux available")
    p.add_argument("--compile-mode", type=str, default="reduce-overhead")

    # io
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--save-every", type=int, default=5)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
