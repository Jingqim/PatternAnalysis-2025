# ISIC 2020 — community 256×256 dataset via kagglehub → PyTorch Datasets (with stratified train/val split)
# Works on Windows/macOS/Linux. Requires: kagglehub, pandas, scikit-learn, pillow, torch, torchvision.
# Assumes you have a Kaggle API key configured for kagglehub authentication.

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import kagglehub  # pip install kagglehub

# -----------------------------
# Config
# -----------------------------
DATA_HANDLE = "ziadloo/isic-2020-256x256"  # community 256×256 pack
IMG_SIZE = 224
VAL_SIZE = 0.2
SEED = 42

# -----------------------------
# Dataset
# -----------------------------
class ISIC256Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, label_col: str, transform=None):
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

# -----------------------------
# Column detection & path resolution
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
    # Collect directories that contain JPG/JPEG/PNGs (shallow + deep)
    dirs = set()
    # Common top-level guesses
    for guess in ["", "train", "train_images", "images", "images/train", "jpeg/train"]:
        p = (base / guess).resolve()
        if p.exists():
            dirs.add(p)

    # Scan recursively for any images and remember their parents
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in base.rglob(ext):
            try:
                dirs.add(p.parent.resolve())
            except Exception:
                dirs.add(p.parent)

    # Return a stable list
    return list(sorted(dirs, key=lambda x: str(x)))

def _resolve_image_path_strings(df: pd.DataFrame, img_col: str, search_roots: List[Path]) -> pd.DataFrame:
    def resolve_one(name: str) -> Optional[str]:
        p = Path(name)
        # If CSV stores absolute/relative full paths
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.exists():
            return str(p)

        stem = Path(name).stem  # tolerate IDs like ISIC_123 with/without extension
        candidates = []
        for root in search_roots:
            candidates.extend([
                root / f"{stem}.jpg",
                root / f"{stem}.jpeg",
                root / f"{stem}.png",
            ])
            # common nested structures
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
    df = df[df["filepath"].notna()]
    return df

# -----------------------------
# Build datasets using kagglehub
# -----------------------------
def build_isic2020_256_datasets(
    data_handle: str = DATA_HANDLE,
    img_size: int = IMG_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
    force_img_col: Optional[str] = None,
    force_label_col: Optional[str] = None,
):
    # 1) Download with kagglehub (returns local cache path)
    data_root = Path(kagglehub.dataset_download(data_handle)).resolve()

    # 2) Find a CSV with labels
    csvs = _find_csv_candidates(data_root)
    if not csvs:
        raise FileNotFoundError("No CSV files found in the downloaded dataset.")
    # Prefer train-like CSV names first
    csvs_sorted = sorted(
        csvs,
        key=lambda p: (("train" not in p.name.lower()), len(p.name))
    )
    csv_path = csvs_sorted[0]
    df = pd.read_csv(csv_path)

    # 3) Choose columns for image + label
    if force_img_col and force_img_col in df.columns:
        img_col = force_img_col
    else:
        img_col = None

    if force_label_col and force_label_col in df.columns:
        label_col = force_label_col
    else:
        label_col = None

    if img_col is None or label_col is None:
        auto_img_col, auto_label_col = _pick_cols(df)
        img_col = img_col or auto_img_col
        label_col = label_col or auto_label_col

    # Map string labels (benign/malignant) -> 0/1 if needed
    if df[label_col].dtype == object:
        mapping = {"benign": 0, "malignant": 1, "negative": 0, "positive": 1}
        df[label_col] = df[label_col].map(lambda x: mapping.get(str(x).lower(), x))

    # Ensure numeric label and drop NaNs
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)

    # 4) Resolve actual file paths from IDs such as isic_id → ISIC_xxx.jpg
    img_dirs = _guess_image_dirs(data_root)
    df = _resolve_image_path_strings(df, img_col, img_dirs)
    if df.empty:
        raise RuntimeError(
            "Could not resolve any image file paths from the CSV. "
            f"Tried roots: {[str(p) for p in img_dirs]}"
        )

    # 5) Stratified split
    y = df[label_col].astype(int)
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=y
    )

    # 6) Transforms
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

    train_ds = ISIC256Dataset(train_df, img_col="filepath", label_col=label_col, transform=train_tfms)
    val_ds   = ISIC256Dataset(val_df,   img_col="filepath", label_col=label_col, transform=val_tfms)
    return train_ds, val_ds

# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    # If your CSV uses 'isic_id' and 'target', you can force them like this:
    # tr, va = build_isic2020_256_datasets(force_img_col="isic_id", force_label_col="target")
    tr, va = build_isic2020_256_datasets()

    print("Train samples:", len(tr))
    print("Val samples:", len(va))

    x, y = tr[0]
    print("One sample:", x.shape, "label:", y.item())
