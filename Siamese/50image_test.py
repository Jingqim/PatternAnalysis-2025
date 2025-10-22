#%% DATA LOADING & PREPROCESSING (uses sklearn only here)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from sklearn.model_selection import GroupShuffleSplit  # preprocessing only

# -------------------------
# Config (hardcoded paths)
# -------------------------
CSV_PATH = Path("./train-metadata.csv")   # columns: isic_id, patient_id, target
IMG_DIR  = Path("./train-image/image")          # files named: <isic_id>.jpg
IMG_SIZE = 224
NUM_WORKERS = 4
BATCH_SIZE  = 64
SEED = 42
TEST_SIZE = 0.20
VAL_SIZE  = 0.10

# -------------------------
# Load CSV & attach file paths (assumes .jpg)
# -------------------------
df = pd.read_csv(CSV_PATH)[["isic_id", "patient_id", "target"]].copy()
df["target"] = df["target"].astype(int)
df["filepath"] = df["isic_id"].apply(lambda x: IMG_DIR / f"{x}.jpg")

exists_mask = df["filepath"].apply(lambda p: Path(p).is_file())
missing = (~exists_mask).sum()
if missing:
    print(f"[info] Dropping {missing} rows with missing image files (*.jpg).")
df = df[exists_mask].reset_index(drop=True)

print(f"[info] Total images found: {len(df)}")
print("[info] Class counts (all):", df["target"].value_counts().to_dict())

# -------------------------
# 3-way split by patient (no leakage) — sklearn only here
# -------------------------
def group_split_three_way(frame: pd.DataFrame,
                          test_size: float = TEST_SIZE,
                          val_size: float  = VAL_SIZE,
                          seed: int = SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(gss1.split(frame, y=frame["target"], groups=frame["patient_id"]))
    df_trainval = frame.iloc[idx_trainval].reset_index(drop=True)
    df_test     = frame.iloc[idx_test].reset_index(drop=True)

    adjusted_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    idx_train, idx_val = next(gss2.split(df_trainval, y=df_trainval["target"], groups=df_trainval["patient_id"]))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val   = df_trainval.iloc[idx_val].reset_index(drop=True)

    # Safety checks
    assert set(df_train.patient_id).isdisjoint(df_val.patient_id)
    assert set(df_train.patient_id).isdisjoint(df_test.patient_id)
    assert set(df_val.patient_id).isdisjoint(df_test.patient_id)
    return df_train, df_val, df_test

train_df, val_df, test_df = group_split_three_way(df, TEST_SIZE, VAL_SIZE, SEED)
print(f"[split] Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
print("[split] Class counts (train):", train_df["target"].value_counts().to_dict())
print("[split] Class counts (val):  ", val_df["target"].value_counts().to_dict())
print("[split] Class counts (test): ", test_df["target"].value_counts().to_dict())

# -------------------------
# Minimal dataset (no external libs)
# -------------------------
class ISIC2020Dataset(Dataset):
    """
    - Reads RGB image via torchvision.io.read_image -> [C,H,W] uint8
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

# (optional) build normal loaders for your classifier workflow
train_loader_cls = DataLoader(ISIC2020Dataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
val_loader_cls   = DataLoader(ISIC2020Dataset(val_df),   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
test_loader_cls  = DataLoader(ISIC2020Dataset(test_df),  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)


#%% TRAINING — SIAMESE (ONE-SHOT) ON FIRST 50 IMAGES (PyTorch only; tqdm for progress)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS    = 5
BATCH_S   = 64
LR        = 1e-3
EMB_DIM   = 128

# -------------------------
# Build a 50-image subset and pair dataset
# -------------------------
subset_df = df.iloc[:50].reset_index(drop=True)
base50_ds = ISIC2020Dataset(subset_df)

class PairDataset(Dataset):
    """
    Produces pairs (x1, x2, same_label):
      same_label = 1.0 if labels match else 0.0
    Also tracks whether the pair includes any positive example to drive sampling weights.
    """
    def __init__(self, base_ds: ISIC2020Dataset):
        self.base = base_ds
        # cache labels (int)
        self.labels = np.array(self.base.df["target"].tolist(), dtype=np.int64)
        # all unordered pairs i<j
        self.pairs = []
        self.same = []
        self.contains_pos = []
        n = len(self.labels)
        for i in range(n):
            for j in range(i+1, n):
                yi, yj = self.labels[i], self.labels[j]
                self.pairs.append((i, j))
                self.same.append(1.0 if yi == yj else 0.0)
                self.contains_pos.append(1.0 if (yi == 1 or yj == 1) else 0.0)
        self.same = torch.tensor(self.same, dtype=torch.float32)
        self.contains_pos = torch.tensor(self.contains_pos, dtype=torch.float32)

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        x1, _ = self.base[i]
        x2, _ = self.base[j]
        return x1, x2, self.same[idx]

pair_ds = PairDataset(base50_ds)

# Weighted sampler: upweight pairs that include at least one positive sample
w = np.ones(len(pair_ds), dtype=np.float32)
w[pair_ds.contains_pos.numpy() == 1.0] *= 8.0
sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

pair_loader = DataLoader(pair_ds, batch_size=BATCH_S, sampler=sampler,
                         num_workers=0, pin_memory=True)

# -------------------------
# Simple Siamese model
# -------------------------
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
        x = self.features(x)          # [B,128,1,1]
        x = x.view(x.size(0), -1)     # [B,128]
        x = self.fc(x)                # [B,emb_dim]
        return F.normalize(x, dim=1)  # L2-normalize for metric learning

class SiameseNet(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.encoder = SmallEncoder(emb_dim)
        self.head = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        diff = torch.abs(z1 - z2)         # simple comparator
        prob_same = self.head(diff).squeeze(1)  # [B]
        return prob_same

model = SiameseNet(EMB_DIM).to(DEVICE)
criterion = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Train loop with tqdm
# -------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(pair_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    running_loss = 0.0
    running_acc  = 0.0
    n_batches = 0

    for x1, x2, y in pbar:
        x1 = x1.to(DEVICE, non_blocking=True)
        x2 = x2.to(DEVICE, non_blocking=True)
        y  = y.to(DEVICE)

        optim.zero_grad()
        probs = model(x1, x2)           # probability "same class"
        loss = criterion(probs, y)
        loss.backward()
        optim.step()

        with torch.no_grad():
            preds = (probs >= 0.5).float()
            acc = (preds == y).float().mean().item()

        n_batches += 1
        running_loss += loss.item()
        running_acc  += acc
        pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}",
                         acc=f"{running_acc/n_batches:.3f}")

    print(f"[epoch {epoch}] loss={running_loss/n_batches:.4f}  acc={running_acc/n_batches:.3f}")

# (Optional) After training, do one-shot inference by encoding a single support example per class
# and comparing a query image embedding to supports via |z_q - z_support|.
