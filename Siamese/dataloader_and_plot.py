# isic2020_three_way_split_preview_no_resolve.py
# Hardcoded layout:
#   ./train-metadata.csv
#   ./train-image/<isic_id>.jpg

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from torchvision.io import read_image  # part of torchvision/PyTorch

# -------------------------
# Config
# -------------------------
CSV_PATH = Path("./train-metadata.csv")
IMG_DIR  = Path("./train-image/image")
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
TEST_SIZE = 0.20  # 20% patients -> test
VAL_SIZE  = 0.10  # 10% patients -> val (of the whole dataset)
SEED = 42


# -------------------------
# Load metadata and attach file paths (assumes .jpg)
# -------------------------
def load_metadata(csv_path: Path, img_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)[["isic_id", "patient_id", "target"]].copy()
    df["target"] = df["target"].astype(int)
    # Directly assume .jpg filenames
    df["filepath"] = df["isic_id"].apply(lambda x: img_dir / f"{x}.jpg")
    # Drop rows where the expected file is missing
    exists_mask = df["filepath"].apply(lambda p: Path(p).is_file())
    missing = (~exists_mask).sum()
    if missing:
        print(f"[info] Dropping {missing} rows with missing image files (*.jpg).")
    df = df[exists_mask].reset_index(drop=True)
    return df

# -------------------------
# 3-way split using scikit-learn (grouped by patient_id)
# -------------------------
def group_split_three_way(df: pd.DataFrame,
                          test_size: float = TEST_SIZE,
                          val_size: float = VAL_SIZE,
                          seed: int = SEED):
    # 1) Hold out TEST by groups (patients)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(gss1.split(df, y=df["target"], groups=df["patient_id"]))
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test     = df.iloc[idx_test].reset_index(drop=True)

    # 2) TRAIN vs VAL from remainder (adjust the fraction)
    adjusted_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    idx_train, idx_val = next(gss2.split(
        df_trainval, y=df_trainval["target"], groups=df_trainval["patient_id"]
    ))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val   = df_trainval.iloc[idx_val].reset_index(drop=True)

    # Sanity: no patient overlap
    assert set(df_train.patient_id).isdisjoint(df_val.patient_id)
    assert set(df_train.patient_id).isdisjoint(df_test.patient_id)
    assert set(df_val.patient_id).isdisjoint(df_test.patient_id)

    return df_train, df_val, df_test

# -------------------------
# Minimal Dataset
# -------------------------
class ISIC2020Dataset(Dataset):
    """
    - Reads image with torchvision.io.read_image -> [C,H,W] uint8 RGB
    - Resizes to IMG_SIZE with F.interpolate
    - Returns float tensor in [0,1] and float label (0.0/1.0)
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

        img = read_image(path)  # uint8 [C,H,W] in RGB
        img = img.to(torch.float32) / 255.0  # [0,1]

        if img.shape[1] != IMG_SIZE or img.shape[2] != IMG_SIZE:
            img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False).squeeze(0)
        return img, torch.tensor(y, dtype=torch.float32)

# -------------------------
# Plot first N images with metadata
# -------------------------
def plot_first_n(df: pd.DataFrame, n: int = 10):
    n = min(n, len(df))
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            row = df.iloc[i]
            img = read_image(str(row["filepath"]))  # [C,H,W] uint8
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(f"{row['isic_id']}\npatient={row['patient_id']}  target={row['target']}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


##%%

# -------------------------
# Main
# -------------------------
def main():
    # 1) Load metadata + file paths
    df = load_metadata(CSV_PATH, IMG_DIR)
    print(f"[info] Total images found: {len(df)}")
    print("[info] Class counts (all):", df["target"].value_counts().to_dict())

    # 2) Patient-wise 3-way split (sklearn)
    train_df, val_df, test_df = group_split_three_way(df, TEST_SIZE, VAL_SIZE, SEED)
    print(f"[split] Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    print("[split] Class counts (train):", train_df["target"].value_counts().to_dict())
    print("[split] Class counts (val):  ", val_df["target"].value_counts().to_dict())
    print("[split] Class counts (test): ", test_df["target"].value_counts().to_dict())

    # 3) Build loaders (ready for training if you need them)
    train_loader = DataLoader(ISIC2020Dataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(ISIC2020Dataset(val_df),   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(ISIC2020Dataset(test_df),  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # 4) Quick batch check
    xb, yb = next(iter(train_loader))
    print("[batch] train:", xb.shape, yb.shape, yb[:8].tolist())

    # 5) Plot first 10 images from the full dataset with metadata
    plot_first_n(df, n=10)

if __name__ == "__main__":
    main()
