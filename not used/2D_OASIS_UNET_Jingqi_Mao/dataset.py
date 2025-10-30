# dataset.py  — PyTorch DataLoader for OASIS 2D PNG slices (images+masks)
# Expected structure (next to this file):
#   ./keras_png_slices_data/
#       keras_png_slices_train/          (e.g., case_001_slice_0.nii.png)
#       keras_png_slices_validate/
#       keras_png_slices_test/
#       keras_png_slices_seg_train/      (e.g., seg_001_slice_0.nii.png)
#       keras_png_slices_seg_validate/
#       keras_png_slices_seg_test/

from pathlib import Path
from typing import List, Tuple
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

# --------------------
# Config (editables)
# --------------------
SIZE_HW = (256, 256)     # (H, W)
BATCH    = 8
NUM_WORKERS = 4
PIN_MEMORY = True

# --------------------
# Pairing helpers
# --------------------
def _collect_all_pngs(folder: Path) -> List[Path]:
    # Accept both *.png and *.nii.png
    return sorted(list(folder.glob("*.png")) + list(folder.glob("*.nii.png")))

def _clean_stem(p: Path) -> str:
    """
    Return filename stem without double suffix.
    'case_001_slice_0.nii.png' -> 'case_001_slice_0'
    'seg_001_slice_0.png'      -> 'seg_001_slice_0'
    """
    s = p.stem            # remove .png
    if s.endswith(".nii"):
        s = Path(s).stem  # remove .nii
    return s

def _to_seg_stem(img_stem: str) -> str:
    # 'case_001_slice_10' -> 'seg_001_slice_10'
    return "seg_" + img_stem[len("case_"):] if img_stem.startswith("case_") else img_stem

def _pair_paths(img_dir: Path, msk_dir: Path) -> List[Tuple[str, str]]:
    img_paths = _collect_all_pngs(img_dir)
    msk_paths = _collect_all_pngs(msk_dir)

    # Map masks by cleaned stem
    msk_by_stem = {_clean_stem(p): str(p) for p in msk_paths}

    pairs: List[Tuple[str, str]] = []
    for ip in img_paths:
        ist = _clean_stem(ip)
        tgt = _to_seg_stem(ist)
        mp = msk_by_stem.get(tgt)
        if mp:
            pairs.append((str(ip), mp))

    if not pairs:
        # Helpful one-time diagnostics
        print("\n--- Pairing diagnostics ---")
        print("Images:", img_dir, "count:", len(img_paths))
        print("Masks :", msk_dir, "count:", len(msk_paths))
        print("Sample images:", [p.name for p in img_paths[:5]])
        print("Sample masks :", [p.name for p in msk_paths[:5]])
        print("---------------------------\n")

    pairs.sort(key=lambda t: os.path.basename(t[0]))
    return pairs

# --------------------
# Tensor helpers
# --------------------
def _zscore(x: torch.Tensor) -> torch.Tensor:
    # x: 1xHxW float
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-6)

def _resize_img(x: torch.Tensor, size_hw=SIZE_HW) -> torch.Tensor:
    # x: 1xHxW float -> bilinear
    x = x.unsqueeze(0)  # 1x1xHxW
    x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return x.squeeze(0) # 1xH'xW'

def _resize_msk(y: torch.Tensor, size_hw=SIZE_HW) -> torch.Tensor:
    # y: 1xHxW uint8/long -> nearest (preserve IDs)
    y = y.unsqueeze(0).float()       # 1x1xHxW
    y = F.interpolate(y, size=size_hw, mode="nearest")
    return y.squeeze(0).long()       # 1xH'xW'

# --------------------
# Dataset
# --------------------
class BrainSeg2D(Dataset):
    """
    Returns:
      img: float tensor 1xH'xW' (z-scored)
      msk: long  tensor H'xW'   (class IDs)
    """
    def __init__(self, img_dir: Path, msk_dir: Path, size_hw=SIZE_HW, augment: bool=False):
        self.size_hw = size_hw
        self.augment = augment
        self.pairs = _pair_paths(img_dir, msk_dir)
        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs under:\n  {img_dir}\n  {msk_dir}\n"
                               "Ensure 'case_…' images match 'seg_…' masks.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]

        # Image: CxHxW uint8 -> 1xHxW float
        img = read_image(img_path)
        if img.shape[0] == 3:
            img = img.float().mean(0, keepdim=True)  # grayscale
        else:
            img = img.float()

        # Mask: CxHxW uint8 -> 1xHxW (take first channel if colored)
        msk = read_image(msk_path)
        if msk.shape[0] > 1:
            msk = msk[:1, ...]

        # Resize (torch)
        img = _resize_img(img, self.size_hw)
        msk = _resize_msk(msk, self.size_hw)

        # Normalize image
        img = _zscore(img)

        # Optional paired aug: horizontal flip
        if self.augment and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            msk = torch.flip(msk, dims=[2])

        return img.contiguous(), msk.squeeze(0).long().contiguous()

# --------------------
# Dataloader factory (anchored to this script’s folder)
# --------------------
def create_dataloaders_here(batch_size=BATCH, size_hw=SIZE_HW, augment_train=True):
    here = Path(__file__).parent.resolve()
    data = here / "keras_png_slices_data"

    img_train = data / "keras_png_slices_train"
    msk_train = data / "keras_png_slices_seg_train"
    img_val   = data / "keras_png_slices_validate"
    msk_val   = data / "keras_png_slices_seg_validate"
    img_test  = data / "keras_png_slices_test"
    msk_test  = data / "keras_png_slices_seg_test"

    ds_train = BrainSeg2D(img_train, msk_train, size_hw=size_hw, augment=augment_train)
    ds_val   = BrainSeg2D(img_val,   msk_val,   size_hw=size_hw, augment=False)
    ds_test  = BrainSeg2D(img_test,  msk_test,  size_hw=size_hw, augment=False)

    ld_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    ld_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    ld_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return ld_train, ld_val, ld_test

# -------------
# Smoke test
# -------------
if __name__ == "__main__":
    tr, va, te = create_dataloaders_here()
    xb, yb = next(iter(tr))
    print("✓ Train batches:", len(tr), "| X", xb.shape, xb.dtype, "| Y", yb.shape, yb.dtype)
    print("Unique labels in first batch:", torch.unique(yb))
