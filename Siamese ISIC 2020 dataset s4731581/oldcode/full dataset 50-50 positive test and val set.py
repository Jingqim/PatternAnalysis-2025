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
    Prints dataset stats once (call from main under __main__ guard).
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

def make_balanced_subset(split_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Build a balanced IMAGE subset from a split: includes ALL positives and
    an equal number of negatives sampled at random from the same split.
    (Keeps patient-wise separation inherited from the split.)
    """
    rng = np.random.default_rng(seed)
    pos_df = split_df[split_df["target"] == 1]
    neg_df = split_df[split_df["target"] == 0]
    k_pos = len(pos_df)
    if k_pos == 0:
        # fallback: no positives available; just return some negatives (rare)
        return neg_df.sample(n=min(1000, len(neg_df)), random_state=seed).reset_index(drop=True)
    k_neg = min(k_pos, len(neg_df))
    neg_sample = neg_df.sample(n=k_neg, random_state=seed)
    out = pd.concat([pos_df, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


#%% TRAINING — Use full train split; balanced val/test subsets; 50 epochs; tqdm everywhere
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- Model ---
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
        return nn.functional.normalize(x, dim=1)  # L2-normalized embeddings

class SiameseNet(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.encoder = SmallEncoder(emb_dim)              # ONE encoder, used twice (shared weights)
        self.head = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())
    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        return self.head(torch.abs(z1 - z2)).squeeze(1)   # prob "same class"

# --- On-the-fly pair dataset for TRAIN (avoids O(N^2) pairs) ---
class OnTheFlyPairs(torch.utils.data.Dataset):
    """
    Samples pairs at access time to avoid quadratic memory.
    - Builds index lists for positives and negatives from a base ISIC2020Dataset.
    - Each __getitem__ randomly chooses SAME vs DIFFERENT with 50/50 probability.
      For SAME: chooses pos-pos (if possible) 50% of the time, else neg-neg.
      For DIFF: chooses pos-neg.
    - 'length' controls how many pairs form an 'epoch'.
    """
    def __init__(self, base_ds, length_multiplier: int = 4, seed: int = 0):
        self.base = base_ds
        labels = np.array(self.base.df["target"].tolist(), dtype=np.int64)
        self.pos_idx = np.where(labels == 1)[0].tolist()
        self.neg_idx = np.where(labels == 0)[0].tolist()
        self.rng = np.random.default_rng(seed)
        self.length = max(1, length_multiplier * len(self.base))  # pairs per epoch

    def __len__(self): return self.length

    def __getitem__(self, _):
        make_same = self.rng.random() < 0.5
        if make_same:
            # try pos-pos half the time (if >=2 pos exist), else neg-neg
            if len(self.pos_idx) >= 2 and self.rng.random() < 0.5:
                i, j = self.rng.choice(self.pos_idx, size=2, replace=False)
            else:
                i, j = self.rng.choice(self.neg_idx, size=2, replace=False)
            y = 1.0
        else:
            # different -> pos vs neg (if no pos, fallback to neg-neg which gives y=1)
            if len(self.pos_idx) >= 1 and len(self.neg_idx) >= 1:
                i = int(self.rng.choice(self.pos_idx))
                j = int(self.rng.choice(self.neg_idx))
                y = 0.0
            else:
                i, j = self.rng.choice(self.neg_idx, size=2, replace=False)
                y = 1.0
        x1, _ = self.base[i]
        x2, _ = self.base[j]
        return x1, x2, torch.tensor(y, dtype=torch.float32)

# --- Balanced pair evaluator for VAL/TEST (caps total pairs for speed) ---
def make_balanced_pair_indices(labels: np.ndarray, max_pairs: int = 10000, seed: int = 0):
    """
    From a label array for an IMAGE subset, return indices of a balanced set of PAIRS:
    equal number of same and different, capped at max_pairs total.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    # Enumerate some pairs but sample to stay under cap
    same_pairs = []
    # pos-pos
    if len(pos) >= 2:
        idx = np.array([(int(pos[a]), int(pos[b])) for a in range(len(pos)) for b in range(a+1, len(pos))], dtype=np.int32)
        same_pairs.append(idx)
    # neg-neg
    if len(neg) >= 2:
        idx = np.array([(int(neg[a]), int(neg[b])) for a in range(len(neg)) for b in range(a+1, len(neg))], dtype=np.int32)
        same_pairs.append(idx)
    same_pairs = np.concatenate(same_pairs, axis=0) if same_pairs else np.empty((0,2), dtype=np.int32)

    # different pairs (pos,neg)
    diff_pairs = np.array([(int(i), int(j)) for i in pos for j in neg], dtype=np.int32)

    if len(same_pairs) == 0 or len(diff_pairs) == 0:
        # degenerate case
        take_s = min(len(same_pairs), max_pairs//2)
        take_d = min(len(diff_pairs), max_pairs//2)
    else:
        half = max_pairs // 2
        take_s = min(len(same_pairs), half)
        take_d = min(len(diff_pairs), half)

    sel_same = same_pairs[rng.choice(len(same_pairs), size=take_s, replace=False)] if take_s > 0 else np.empty((0,2), dtype=np.int32)
    sel_diff = diff_pairs[rng.choice(len(diff_pairs), size=take_d, replace=False)] if take_d > 0 else np.empty((0,2), dtype=np.int32)
    # stack and shuffle
    pairs = np.vstack([sel_same, sel_diff]) if (take_s + take_d) > 0 else np.empty((0,2), dtype=np.int32)
    rng.shuffle(pairs)
    # labels: same=1, diff=0
    y_same = np.concatenate([np.ones(take_s, dtype=np.float32), np.zeros(take_d, dtype=np.float32)]) if (take_s + take_d) > 0 else np.array([], dtype=np.float32)
    return pairs, y_same

class FixedPairsDataset(torch.utils.data.Dataset):
    """Holds a fixed list of PAIRS (i,j) over a base image dataset + pair labels."""
    def __init__(self, base_ds, pair_idx: np.ndarray, pair_y: np.ndarray):
        self.base = base_ds
        self.pairs = pair_idx
        self.y = pair_y.astype(np.float32)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, k):
        i, j = self.pairs[k]
        x1, _ = self.base[int(i)]
        x2, _ = self.base[int(j)]
        return x1, x2, torch.tensor(self.y[k], dtype=torch.float32)

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
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT = True  # keep workers alive across epochs (fewer respawns)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    print(f"[info] Using device: {DEVICE}")

    # --- Prepare patient-wise splits (runs ONCE; not in workers) ---
    from __main__ import prepare_splits, ISIC2020Dataset, make_balanced_subset
    train_df, val_df, test_df = prepare_splits(CSV_PATH, IMG_DIR)

    # --- Use ENTIRE TRAIN split for training images ---
    train_ds = ISIC2020Dataset(train_df)

    # --- Balanced IMAGE subsets for VAL & TEST (50% pos, 50% neg by image count) ---
    val_bal_df  = make_balanced_subset(val_df,  seed=43)
    test_bal_df = make_balanced_subset(test_df, seed=44)
    print("Balanced val images:", len(val_bal_df), "positives:", int((val_bal_df['target']==1).sum()),
          "negatives:", int((val_bal_df['target']==0).sum()))
    print("Balanced test images:", len(test_bal_df), "positives:", int((test_bal_df['target']==1).sum()),
          "negatives:", int((test_bal_df['target']==0).sum()))

    val_ds  = ISIC2020Dataset(val_bal_df)
    test_ds = ISIC2020Dataset(test_bal_df)

    # --- TRAIN: on-the-fly pair sampling over FULL train_ds ---
    # length_multiplier decides pairs per epoch (e.g., 4x images)
    train_pair_ds = OnTheFlyPairs(train_ds, length_multiplier=4, seed=123)

    # --- VAL/TEST: build balanced PAIR sets capped by max pairs for speed ---
    MAX_EVAL_PAIRS = 10000  # adjust if you want more/less eval pairs
    val_pairs_idx, val_pairs_y   = make_balanced_pair_indices(val_bal_df["target"].to_numpy(),  max_pairs=MAX_EVAL_PAIRS, seed=1)
    test_pairs_idx, test_pairs_y = make_balanced_pair_indices(test_bal_df["target"].to_numpy(), max_pairs=MAX_EVAL_PAIRS, seed=2)

    val_pair_ds  = FixedPairsDataset(val_ds,  val_pairs_idx,  val_pairs_y)
    test_pair_ds = FixedPairsDataset(test_ds, test_pairs_idx, test_pairs_y)

    # --- DataLoaders (Windows-safe; workers only created in __main__) ---
    train_loader = DataLoader(train_pair_ds, batch_size=64, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)
    val_loader   = DataLoader(val_pair_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)
    test_loader  = DataLoader(test_pair_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)

    # --- Model / loss / optim ---
    model = SiameseNet(EMB_DIM).to(DEVICE)
    bce   = nn.BCELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Train for 50 epochs; validate each epoch with tqdm ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
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
