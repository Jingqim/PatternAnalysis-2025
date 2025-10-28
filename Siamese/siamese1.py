# siamese_isic2020_256_cuda.py
# -----------------------------------------------------------
# Siamese network for ISIC-2020 (community 256×256 via kagglehub)
# - CUDA-ready (uses GPU if available), prints CUDA and GPU info
# - Mixed precision training (torch.cuda.amp) for speed
# - Saves per-epoch training accuracy (pair classification) and loss
# - Saves per-epoch validation accuracy (prototype-based), confusion matrix
# - Logs to ./runs_siamese_isic256_<timestamp>/metrics.csv
#
# Install (same env):
#   pip install kagglehub pandas scikit-learn pillow torch torchvision
#
# Run:
#   python siamese_isic2020_256_cuda.py
# -----------------------------------------------------------

import os
import csv
import time
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms as T, models

import kagglehub  # pip install kagglehub

# -----------------------------
# Config
# -----------------------------
DATA_HANDLE = "ziadloo/isic-2020-256x256"  # community 256×256 pack
VAL_SIZE = 0.2
IMG_SIZE = 224
SEED = 42

# Training
BATCH_SIZE = 64                # pairs per batch
EPOCHS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2                # if Windows errors, set to 0
SAVE_EVERY = 2                 # checkpoint every N epochs

# Output dir with timestamp
RUN_DIR = Path(f"./runs_siamese_isic256_{time.strftime('%Y%m%d_%H%M%S')}")
RUN_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True  # faster on GPU for fixed image sizes

# -----------------------------
# Dataset utils (auto-detect CSV & resolve filepaths)
# -----------------------------
IMG_COL_CANDIDATES = [
    "image_name", "image", "file_name", "filename", "id", "Image", "img",
    "isic_id", "image_id"
]
LABEL_COL_CANDIDATES = [
    "target", "label", "Labels", "class", "benign_malignant", "malignant"
]

def _find_csv_candidates(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.csv") if p.is_file()]

def _pick_cols(df: pd.DataFrame) -> Tuple[str, str]:
    img_col = next((c for c in IMG_COL_CANDIDATES if c in df.columns), None)
    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    if img_col is None or label_col is None:
        raise ValueError(
            "Could not identify image/label columns. "
            f"Have columns: {list(df.columns)}. "
            f"Expected one of {IMG_COL_CANDIDATES} for image and {LABEL_COL_CANDIDATES} for label."
        )
    return img_col, label_col

def _guess_image_dirs(base: Path) -> List[Path]:
    dirs = set()
    for guess in ["", "train", "train_images", "images", "images/train", "jpeg/train"]:
        p = (base / guess).resolve()
        if p.exists():
            dirs.add(p)
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in base.rglob(ext):
            try:
                dirs.add(p.parent.resolve())
            except Exception:
                dirs.add(p.parent)
    return list(sorted(dirs, key=lambda x: str(x)))

def _resolve_image_paths(df: pd.DataFrame, img_col: str, search_roots: List[Path]) -> pd.DataFrame:
    def resolve_one(name: str) -> Optional[str]:
        p = Path(name)
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.exists():
            return str(p)
        stem = Path(name).stem
        candidates = []
        for root in search_roots:
            candidates.extend([
                root / f"{stem}.jpg",
                root / f"{stem}.jpeg",
                root / f"{stem}.png",
            ])
            for sub in ("images", "train", "train_images", "jpeg/train"):
                candidates.extend([
                    root / sub / f"{stem}.jpg",
                    root / sub / f"{stem}.jpeg",
                    root / sub / f"{stem}.png",
                ])
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    df = df.copy()
    df["filepath"] = df[img_col].astype(str).apply(resolve_one)
    df = df[df["filepath"].notna()].copy()
    return df

# -----------------------------
# Image dataset (single image, label)
# -----------------------------
class ISIC256Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str = "filepath", label_col: str = "target", transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row[self.img_col])
        label = int(row[self.label_col])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, torch.tensor(label, dtype=torch.long)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

# -----------------------------
# Pair dataset (for Siamese training)
# -----------------------------
class PairDataset(Dataset):
    """Yields (img1, img2, same_flag) with ~50/50 pos/neg sampling."""
    def __init__(self, base_ds: ISIC256Dataset, transform_pair=None):
        self.base = base_ds
        self.transform = transform_pair
        self.df = base_ds.get_dataframe()
        self.idx_by_label = {
            0: self.df.index[self.df["target"] == 0].tolist(),
            1: self.df.index[self.df["target"] == 1].tolist(),
        }
        assert len(self.idx_by_label[0]) > 0 and len(self.idx_by_label[1]) > 0, "Need both classes in train set."

    def __len__(self) -> int:
        return len(self.base)

    def _load(self, idx: int):
        img_path = self.df.loc[idx, "filepath"]
        label = int(self.df.loc[idx, "target"])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        return im, label

    def __getitem__(self, _):
        is_pos = random.random() < 0.5
        if is_pos:
            y = random.choice([0, 1])
            choices = self.idx_by_label[y]
            if len(choices) >= 2:
                i1, i2 = random.sample(choices, 2)
            else:
                i1 = i2 = choices[0]
        else:
            y1, y2 = (0, 1) if random.random() < 0.5 else (1, 0)
            i1 = random.choice(self.idx_by_label[y1])
            i2 = random.choice(self.idx_by_label[y2])

        im1, lab1 = self._load(i1)
        im2, lab2 = self._load(i2)
        same = int(lab1 == lab2)

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return im1, im2, torch.tensor(same, dtype=torch.float32)

# -----------------------------
# Siamese Network
# -----------------------------
class ResNet18Encoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        m = models.resnet18(weights=weights)  # set weights=None (no internet needed)
        self.features = nn.Sequential(*list(m.children())[:-1])  # global pool -> [B,512,1,1]
        self.out_dim = 512

    def forward(self, x):
        z = self.features(x)        # [B,512,1,1]
        z = z.view(z.size(0), -1)   # [B,512]
        z = F.normalize(z, dim=1)   # cosine-friendly
        return z

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder(weights=None)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))  # scale cosine to logits

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        cos = F.cosine_similarity(e1, e2, dim=1)   # [-1,1]
        logits = self.logit_scale * cos
        return logits, e1, e2

    def embed(self, x):
        return self.encoder(x)

# -----------------------------
# Build datasets (download via kagglehub)
# -----------------------------
def build_isic2020_256(
    data_handle: str = DATA_HANDLE,
    val_size: float = VAL_SIZE,
    img_size: int = IMG_SIZE,
    seed: int = SEED,
):
    data_root = Path(kagglehub.dataset_download(data_handle)).resolve()
    csvs = _find_csv_candidates(data_root)
    if not csvs:
        raise FileNotFoundError("No CSV files found in the downloaded dataset.")
    csvs_sorted = sorted(csvs, key=lambda p: (("train" not in p.name.lower()), len(p.name)))
    csv_path = csvs_sorted[0]
    df = pd.read_csv(csv_path)

    img_col, label_col = _pick_cols(df)
    if df[label_col].dtype == object:
        mapping = {"benign": 0, "malignant": 1, "negative": 0, "positive": 1}
        df[label_col] = df[label_col].map(lambda x: mapping.get(str(x).lower(), x))
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)

    img_dirs = _guess_image_dirs(data_root)
    df = _resolve_image_paths(df, img_col, img_dirs)
    if df.empty:
        raise RuntimeError("Failed to resolve image paths. Check dataset layout.")

    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    y = df[label_col].astype(int)
    tr_df, va_df = train_test_split(df, test_size=val_size, random_state=seed, stratify=y)

    train_img_ds = ISIC256Dataset(tr_df, img_col="filepath", label_col=label_col, transform=train_tfms)
    val_img_ds   = ISIC256Dataset(va_df, img_col="filepath", label_col=label_col, transform=val_tfms)

    # Pair dataset for training (applies same augs to both views independently)
    pair_tfms = train_tfms
    train_pair_ds = PairDataset(train_img_ds, transform_pair=pair_tfms)

    return train_pair_ds, train_img_ds, val_img_ds, data_root

# -----------------------------
# Prototypes & evaluation
# -----------------------------
@torch.no_grad()
def build_class_prototypes(encoder: nn.Module, ds: ISIC256Dataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    feats0, feats1 = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = encoder(x)  # normalized
        feats0.append(z[y == 0])
        feats1.append(z[y == 1])
    z0 = torch.cat(feats0, dim=0).mean(dim=0)
    z1 = torch.cat(feats1, dim=0).mean(dim=0)
    z0 = F.normalize(z0, dim=0)
    z1 = F.normalize(z1, dim=0)
    return z0, z1

@torch.no_grad()
def eval_by_prototypes(model: SiameseNet, val_ds: ISIC256Dataset, proto0: torch.Tensor, proto1: torch.Tensor, device: torch.device):
    loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model.embed(x)
        s0 = F.cosine_similarity(z, proto0.unsqueeze(0).expand_as(z), dim=1)
        s1 = F.cosine_similarity(z, proto1.unsqueeze(0).expand_as(z), dim=1)
        pred = (s1 > s0).long().cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(y.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, digits=4, labels=[0, 1], target_names=["class0", "class1"])
    return acc, cm, report

# -----------------------------
# Training
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA version (PyTorch):", torch.version.cuda)
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    (RUN_DIR / "checkpoints").mkdir(exist_ok=True, parents=True)
    metrics_csv = RUN_DIR / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_pair_acc", "val_acc", "tn", "fp", "fn", "tp", "lr", "secs"])

    train_pair_ds, train_img_ds, val_img_ds, data_root = build_isic2020_256(
        data_handle=DATA_HANDLE, val_size=VAL_SIZE, img_size=IMG_SIZE, seed=SEED
    )
    print("Dataset root:", data_root)
    print("Train singles:", len(train_img_ds), "| Val singles:", len(val_img_ds))
    print("Train pairs/epoch:", len(train_pair_ds))

    train_loader = DataLoader(
        train_pair_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=True
    )

    model = SiameseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    bce = nn.BCEWithLogitsLoss()
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct_pairs = 0
        total_pairs = 0

        for (x1, x2, same) in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            same = same.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _, _ = model(x1, x2)     # [B]
                loss = bce(logits, same)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x1.size(0)

            # training pair accuracy (same/different)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_pairs += (preds == same).sum().item()
            total_pairs += same.numel()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        train_pair_acc = correct_pairs / max(1, total_pairs)
        dt = time.time() - t0

        # ---- Validation via prototypes (single-image classification) ----
        model.eval()
        proto0, proto1 = build_class_prototypes(model.embed, train_img_ds, device)
        val_acc, cm, report = eval_by_prototypes(model, val_img_ds, proto0, proto1, device)

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={epoch_loss:.4f} | "
              f"train_pair_acc={train_pair_acc:.4f} | val_acc={val_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.6f} | {dt:.1f}s")
        print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)
        print(report)

        # Log metrics to CSV
        tn, fp, fn, tp = (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{epoch_loss:.6f}", f"{train_pair_acc:.6f}", f"{val_acc:.6f}",
                             int(tn), int(fp), int(fn), int(tp),
                             f"{optimizer.param_groups[0]['lr']:.8f}", f"{dt:.2f}"])

        # Save best & periodic checkpoints
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val_acc": best_acc,
                    "proto0": proto0.detach().cpu(),
                    "proto1": proto1.detach().cpu(),
                    "IMG_SIZE": IMG_SIZE,
                    "VAL_SIZE": VAL_SIZE,
                },
                RUN_DIR / "checkpoints" / "best.pt",
            )
        if epoch % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                RUN_DIR / "checkpoints" / f"epoch{epoch:02d}.pt",
            )

    print(f"Training done. Best val acc: {best_acc:.4f}")
    print(f"Logs: {RUN_DIR.resolve() / 'metrics.csv'}")
    print(f"Checkpoints: {(RUN_DIR / 'checkpoints').resolve()}")

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    # If you see worker issues on Windows, set NUM_WORKERS = 0 near the top.
    train()
