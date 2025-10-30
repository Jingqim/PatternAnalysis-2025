# dataset.py
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split, GroupShuffleSplit

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------- file discovery ----------
def find_image_dir(root: Path) -> Tuple[Optional[Path], int]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    best, best_n = None, 0
    for p in root.rglob("*"):
        if p.is_dir():
            n = sum(1 for x in p.glob("*") if x.suffix in exts)
            if n > best_n:
                best, best_n = p, n
    return best, best_n


def pick_csv(root: Path) -> Optional[Path]:
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        return None

    def score(p: Path):
        try:
            hdr = pd.read_csv(p, nrows=5, low_memory=False)
        except Exception:
            return (-1, p)
        cols = set(hdr.columns)
        label_like = any(c in cols for c in ["target", "label", "labels", "benign_malignant", "malignant", "is_malignant"])
        id_like = any(c in cols for c in ["isic_id", "image_name", "image", "filename", "file_name", "id", "name"])
        return (int(label_like) + int(id_like), p)

    scored = sorted((score(p) for p in csvs), reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else csvs[0]


def to_isic_stem(s: str) -> Optional[str]:
    m = re.search(r"(ISIC_\d+)", str(s))
    return m.group(1) if m else None


def index_files_by_stem(img_dir: Path):
    from collections import defaultdict
    by = defaultdict(list)
    for p in img_dir.glob("*"):
        if p.is_file():
            m = re.search(r"(ISIC_\d+)", p.stem)
            if m:
                by[m.group(1)].append(p)
    return by


def make_balanced_frame(df: pd.DataFrame, label_col: str = "target", seed: int = 42) -> pd.DataFrame:
    """Undersample the majority class to return a class-balanced subset."""
    vc = df[label_col].value_counts()
    if 0 not in vc or 1 not in vc:
        raise SystemExit("Cannot build a balanced test set because one class is missing.")
    n = int(min(vc[0], vc[1]))
    df0 = df[df[label_col] == 0].sample(n=n, random_state=seed)
    df1 = df[df[label_col] == 1].sample(n=n, random_state=seed)
    return pd.concat([df0, df1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------- datasets ----------
class ISIC2020Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transforms=None):
        self.df = frame.reset_index(drop=True)
        self.t = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        if self.t: img = self.t(img)
        return img, int(r["target"])


class PairDataset(Dataset):
    """Pairs built on the fly: y=1 for same class (PP/NN), y=0 for PN.
       Supports PN/PP/NN mix schedule and optional hard-negative mining."""
    def __init__(self, base_ds: ISIC2020Dataset, pn_pp_nn=(0.6, 0.2, 0.2), length=None):
        import random
        self.base = base_ds
        self.pn, self.pp, self.nn = pn_pp_nn
        self.length = length or len(base_ds)
        labels = self.base.df["target"].astype(int).tolist()
        self.pos_idx = [i for i, y in enumerate(labels) if y == 1]
        self.neg_idx = [i for i, y in enumerate(labels) if y == 0]
        assert self.pos_idx and self.neg_idx, "Both classes required."
        self.hard_negatives: Dict[int, List[int]] = {}
        self.hnm_frac: float = 0.0

    def set_mix(self, pn: float, pp: float, nn: float):
        s = pn + pp + nn
        self.pn, self.pp, self.nn = pn / s, pp / s, nn / s

    def set_hard_negatives(self, pool: Dict[int, List[int]], frac: float):
        self.hard_negatives = pool or {}
        self.hnm_frac = float(frac)

    def __len__(self): return self.length

    def __getitem__(self, _):
        import random
        r = random.random()
        if r < self.pn:
            i = random.choice(self.pos_idx)
            use_hard = (self.hard_negatives and random.random() < self.hnm_frac
                        and i in self.hard_negatives and len(self.hard_negatives[i]) > 0)
            j = random.choice(self.hard_negatives[i]) if use_hard else random.choice(self.neg_idx)
            y = 0
        elif r < self.pn + self.pp:
            i = random.choice(self.pos_idx); j = random.choice(self.pos_idx); y = 1
        else:
            i = random.choice(self.neg_idx); j = random.choice(self.neg_idx); y = 1
        x1, _ = self.base[i]; x2, _ = self.base[j]
        return x1, x2, y


# ---------- split + transforms ----------
def build_datasets(args):
    # 1) root
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
    else:
        import kagglehub
        print("Downloading mirror with kagglehub ...")
        root = Path(kagglehub.dataset_download("ziadloo/isic-2020-256x256")).resolve()
    print("Dataset root:", root)

    # 2) locate images/csv
    img_dir, n_imgs = find_image_dir(root)
    if not img_dir or n_imgs == 0:
        raise SystemExit("Could not locate image files.")
    csv_path = pick_csv(root)
    if not csv_path:
        raise SystemExit("Could not find a CSV with labels.")
    print(f"IMG_DIR: {img_dir} (files: {n_imgs})")
    print(f"CSV: {csv_path}")

    # 3) read + normalize columns
    df = pd.read_csv(csv_path, low_memory=False)
    cand_id = ["isic_id", "image_name", "image", "image_id", "filename", "file_name", "id", "name"]
    id_col = next((c for c in cand_id if c in df.columns), None)
    if id_col is None:
        for c in df.columns:
            sample = df[c].astype(str).head(50).tolist()
            if sum(bool(re.search(r"ISIC_\d+", s)) for s in sample) >= 10:
                id_col = c; break
    if id_col is None:
        raise SystemExit("Could not identify image id column.")

    if "target" in df.columns:
        df["target"] = df["target"].astype(int)
    elif "benign_malignant" in df.columns:
        df["target"] = (df["benign_malignant"].astype(str).str.lower() == "malignant").astype(int)
    elif "label" in df.columns:
        df["target"] = df["label"].astype(int)
    else:
        raise SystemExit("No label column found (target/benign_malignant/label).")

    df["image_stem"] = df[id_col].astype(str).map(to_isic_stem)

    # 4) map to file paths
    by_stem = index_files_by_stem(img_dir)
    df["image_path"] = df["image_stem"].map(lambda s: str(by_stem[s][0]) if s in by_stem else None)
    missing = df["image_path"].isna().sum()
    print(f"Mapped images: {len(df)-missing}/{len(df)} (missing={missing})")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # 5) split (group-aware if patient_id exists)
    val_size, test_size, rnd = args.val_size, args.test_size, args.seed
    if "patient_id" in df.columns:
        print("Using patient-level, group-preserving split.")
        groups = df["patient_id"].astype(str)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rnd)
        idx = np.arange(len(df))
        trval_idx, test_idx = next(gss1.split(idx, groups=groups))
        df_trval = df.iloc[trval_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        val_prop = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=rnd)
        idx2 = np.arange(len(df_trval))
        tr_idx, va_idx = next(gss2.split(idx2, groups=df_trval["patient_id"].astype(str)))
        train_df = df_trval.iloc[tr_idx].reset_index(drop=True)
        val_df = df_trval.iloc[va_idx].reset_index(drop=True)
    else:
        df_trval, df_test = train_test_split(
            df, test_size=test_size, stratify=df["target"].astype(int), random_state=rnd
        )
        val_prop = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            df_trval, test_size=val_prop, stratify=df_trval["target"].astype(int), random_state=rnd
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

    # balanced test (undersampled)
    test_bal_df = make_balanced_frame(df_test, label_col="target", seed=rnd)

    # 6) transforms + datasets
    img_size = args.img_size
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    eval_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(), T.Normalize(mean, std),
    ])

    ds_train = ISIC2020Dataset(train_df, train_tfms)
    ds_val = ISIC2020Dataset(val_df, eval_tfms)
    ds_test_bal = ISIC2020Dataset(test_bal_df, eval_tfms)

    print(f"Datasets ready ✓  train: {len(ds_train)}  val: {len(ds_val)}  test: {len(df_test)}  test_bal: {len(ds_test_bal)}")
    print("Class balance →",
          f"train: {train_df['target'].mean():.3f} pos",
          f"val: {val_df['target'].mean():.3f} pos",
          f"test: {df_test['target'].mean():.3f} pos",
          f"test_bal: {test_bal_df['target'].mean():.3f} pos")

    return ds_train, ds_val, ds_test_bal
