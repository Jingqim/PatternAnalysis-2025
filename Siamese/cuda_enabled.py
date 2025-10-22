#%% DATA LOADING & PREPROCESSING (helpers only; NO side effects at import)
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from sklearn.model_selection import GroupShuffleSplit  # preprocessing only

IMG_SIZE = 224  # shared image size

class ISIC2020Dataset(Dataset):
    """
    - Reads RGB via torchvision.io.read_image -> [C,H,W] uint8
    - Converts to float [0,1], resizes to IMG_SIZE, returns (image, label_float)
    """
    def __init__(self, frame: pd.DataFrame):
        self.df = frame.reset_index(drop=True)
        self.paths = self.df["filepath"].astype(str).tolist()
        self.targets = self.df["target"].astype(int).to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y = float(self.targets[idx])
        img = read_image(path)  # [C,H,W] uint8 RGB
        img = img.to(torch.float32) / 255.0
        if img.shape[1] != IMG_SIZE or img.shape[2] != IMG_SIZE:
            img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False).squeeze(0)
        return img, torch.tensor(y, dtype=torch.float32)

def prepare_splits(csv_path: Path, img_dir: Path,
                   test_size: float = 0.20, val_size: float = 0.10, seed: int = 42):
    """
    Loads metadata, attaches file paths (assumes .jpg), drops missing,
    and returns patient-wise train/val/test splits (no leakage).
    Prints dataset stats once (called from main under __main__ guard).
    """
    df = pd.read_csv(csv_path)[["isic_id", "patient_id", "target"]].copy()
    df["target"] = df["target"].astype(int)
    df["filepath"] = df["isic_id"].apply(lambda x: img_dir / f"{x}.jpg")
    df = df[df["filepath"].apply(lambda p: Path(p).is_file())].reset_index(drop=True)

    print(f"[info] Total images found: {len(df)}")
    print("[info] Class counts (all):", df["target"].value_counts().to_dict())

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(gss1.split(df, y=df["target"], groups=df["patient_id"]))
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test     = df.iloc[idx_test].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed)
    idx_train, idx_val = next(gss2.split(df_trainval, y=df_trainval["target"], groups=df_trainval["patient_id"]))
    train_df = df_trainval.iloc[idx_train].reset_index(drop=True)
    val_df   = df_trainval.iloc[idx_val].reset_index(drop=True)

    print(f"[split] Train={len(train_df)}  Val={len(val_df)}  Test={len(df_test)}")
    print("[split] Class counts (train):", train_df["target"].value_counts().to_dict())
    print("[split] Class counts (val):  ", val_df["target"].value_counts().to_dict())
    print("[split] Class counts (test): ", df_test["target"].value_counts().to_dict())
    return train_df, val_df, df_test







#%% TRAINING — Siamese one-shot on train50; validate on val50; test on test50 (tqdm everywhere)
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

# --- Model & pair dataset definitions (safe at import) ---
EMB_DIM = 128

class SmallEncoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, emb_dim)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=1)  # L2 normalized embeddings

class SiameseNet(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.encoder = SmallEncoder(emb_dim)              # ONE encoder, used twice (shared weights)
        self.head = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())
    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        return self.head(torch.abs(z1 - z2)).squeeze(1)   # prob "same class"

class PairDataset(torch.utils.data.Dataset):
    """
    Builds all unordered pairs (i<j) from a base ISIC2020Dataset.
    Returns (x1, x2, same_label) where same_label=1 if labels match, else 0.
    Also stores contains_pos (any of the two is positive) for sampling weights.
    """
    def __init__(self, base_ds):
        self.base = base_ds
        labels = np.array(self.base.df["target"].tolist(), dtype=np.int64)
        self.pairs, self.same, self.contains_pos = [], [], []
        n = len(labels)
        for i in range(n):
            for j in range(i+1, n):
                yi, yj = labels[i], labels[j]
                self.pairs.append((i, j))
                self.same.append(1.0 if yi == yj else 0.0)
                self.contains_pos.append(1.0 if (yi == 1 or yj == 1) else 0.0)
        self.same = torch.tensor(self.same, dtype=torch.float32)
        self.contains_pos = torch.tensor(self.contains_pos, dtype=torch.float32)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        x1, _ = self.base[i]
        x2, _ = self.base[j]
        return x1, x2, self.same[idx]

# --- small utilities ---
def pick_50_with_k_pos(split_df: pd.DataFrame, k_pos=5, seed=42):
    """Pick 50 samples with ~k_pos positives (if available) from a split."""
    pos = split_df[split_df.target == 1]
    neg = split_df[split_df.target == 0]
    k_pos = min(k_pos, len(pos))
    k_neg = max(0, 50 - k_pos)
    pos_sample = pos.sample(n=k_pos, random_state=seed) if k_pos > 0 else pos.iloc[:0]
    neg_sample = neg.sample(n=min(k_neg, len(neg)), random_state=seed)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

def eval_on_pairs(model, loader, device, desc="Eval", leave=False):
    """Evaluate with tqdm progress; returns (mean_loss, mean_acc)."""
    model.eval()
    bce = nn.BCELoss()
    totL = totA = n = 0
    for x1, x2, y in tqdm(loader, desc=desc, leave=leave):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y  = y.to(device)
        with torch.no_grad():
            p = model(x1, x2)
            L = bce(p, y)
            acc = ((p >= 0.5).float() == y).float().mean().item()
        totL += L.item(); totA += acc; n += 1
    return totL / max(1, n), totA / max(1, n)

def main():
    # --- Config (paths & loader params) ---
    CSV_PATH = Path("./train-metadata.csv")
    IMG_DIR  = Path("./train-image/image")
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT = True  # keep workers alive across epochs (fewer respawns)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    print(f"[info] Using device: {DEVICE}")

    # --- Prepare patient-wise splits (runs ONCE; not in workers) ---
    from __main__ import prepare_splits, ISIC2020Dataset  # safe; both defined above
    train_df, val_df, test_df = prepare_splits(CSV_PATH, IMG_DIR)

    # --- Build 50-image subsets (each: ~5 pos, 45 neg if available) ---
    train50_df = pick_50_with_k_pos(train_df, k_pos=5, seed=42)
    val50_df   = pick_50_with_k_pos(val_df,   k_pos=5, seed=43)
    test50_df  = pick_50_with_k_pos(test_df,  k_pos=5, seed=44)
    print("Subset class counts (train50):", train50_df["target"].value_counts().to_dict())
    print("Subset class counts (val50):  ", val50_df["target"].value_counts().to_dict())
    print("Subset class counts (test50): ", test50_df["target"].value_counts().to_dict())

    # --- Datasets & pair datasets ---
    train50_ds = ISIC2020Dataset(train50_df)
    val50_ds   = ISIC2020Dataset(val50_df)
    test50_ds  = ISIC2020Dataset(test50_df)

    train_pair = PairDataset(train50_ds)
    val_pair   = PairDataset(val50_ds)
    test_pair  = PairDataset(test50_ds)

    # --- Weighted sampler (upsample pairs that include a positive) ---
    w = np.ones(len(train_pair), dtype=np.float32)
    w[train_pair.contains_pos.numpy() == 1.0] *= 8.0
    train_sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    # --- DataLoaders (Windows-safe; workers only created in __main__) ---
    train_loader = DataLoader(train_pair, batch_size=64, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)
    val_loader   = DataLoader(val_pair, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)
    test_loader  = DataLoader(test_pair, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)

    # --- Model / loss / optim ---
    model = SiameseNet(EMB_DIM).to(DEVICE)
    bce   = nn.BCELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Train with tqdm; validate each epoch with tqdm ---
    for epoch in range(1, 6):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/5", leave=False)
        runL = runA = n = 0
        for x1, x2, y in pbar:
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)
            y  = y.to(DEVICE)

            opt.zero_grad()
            p = model(x1, x2)
            L = bce(p, y)
            L.backward()
            opt.step()

            with torch.no_grad():
                acc = ((p >= 0.5).float() == y).float().mean().item()

            n += 1
            runL += L.item()
            runA += acc
            pbar.set_postfix(loss=f"{runL/n:.4f}", acc=f"{runA/n:.3f}")

        vL, vA = eval_on_pairs(model, val_loader, DEVICE, desc=f"Validate (epoch {epoch})", leave=False)
        print(f"[epoch {epoch}] train_loss={runL/n:.4f} train_acc={runA/n:.3f} | val_loss={vL:.4f} val_acc={vA:.3f}")

    # --- Final test with tqdm ---
    tL, tA = eval_on_pairs(model, test_loader, DEVICE, desc="Test", leave=True)
    print(f"[test] loss={tL:.4f} acc={tA:.3f}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Windows-safe for DataLoader workers
    main()
