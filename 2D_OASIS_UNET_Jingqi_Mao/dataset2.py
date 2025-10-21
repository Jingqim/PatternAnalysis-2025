# dataset.py — OASIS 2D PNG dataloader (Torch/TorchVision only)
# Place this file next to: ./keras_png_slices_data/

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
SIZE_HW = (256, 256)  # (H, W)
BATCH = 8

# Tune these as you like; Windows prefers smaller worker counts
NUM_WORKERS = max(2, (os.cpu_count() or 4) // 2)
PIN_MEMORY = True

# --------------------
# Pairing helpers
# --------------------
def _collect_all_pngs(folder: Path) -> List[Path]:
    # Accept both *.png and *.nii.png
    return sorted(list(folder.glob("*.png")) + list(folder.glob("*.nii.png")))

def _clean_stem(p: Path) -> str:
    """'case_001_slice_0.nii.png' -> 'case_001_slice_0' (strip .png and optional .nii)"""
    s = p.stem
    if s.endswith(".nii"):
        s = Path(s).stem
    return s

def _to_seg_stem(img_stem: str) -> str:
    # case_001_slice_10 -> seg_001_slice_10
    return "seg_" + img_stem[len("case_"):] if img_stem.startswith("case_") else img_stem

def _pair_paths(img_dir: Path, msk_dir: Path) -> List[Tuple[str, str]]:
    img_paths = _collect_all_pngs(img_dir)
    msk_paths = _collect_all_pngs(msk_dir)
    msk_by_stem = {_clean_stem(p): str(p) for p in msk_paths}

    pairs: List[Tuple[str, str]] = []
    for ip in img_paths:
        ist = _clean_stem(ip)
        tgt = _to_seg_stem(ist)
        mp = msk_by_stem.get(tgt)
        if mp:
            pairs.append((str(ip), mp))

    if not pairs:
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
    # x: 1xHxW float -> bilinear resize
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return x.squeeze(0)

def _resize_msk_indices(y: torch.Tensor, size_hw=SIZE_HW) -> torch.Tensor:
    # y: HxW long -> nearest resize (preserve IDs)
    y = y.unsqueeze(0).unsqueeze(0).float()  # 1x1xH xW
    y = F.interpolate(y, size=size_hw, mode="nearest")
    return y.squeeze(0).squeeze(0).long()

def read_mask_grayscale_as_indices(msk_path: str, size_hw=SIZE_HW) -> torch.Tensor:
    """
    TorchVision-only: read a grayscale PNG and map its unique gray values to
    contiguous class IDs 0..K-1. Works whether values are {0,1,2,3} or {0,85,170,255}.
    """
    m = read_image(msk_path)  # CxHxW, uint8
    if m.shape[0] > 1:
        m = m[:1, ...]        # take first channel if RGB slipped in
    m = m.squeeze(0)          # HxW, uint8

    flat = m.view(-1)
    uniq, inv = torch.unique(flat, sorted=True, return_inverse=True)  # uniq: K, inv: N
    m_idx = inv.view_as(m).long()  # HxW, class IDs 0..K-1

    m_idx = _resize_msk_indices(m_idx, size_hw=size_hw)
    return m_idx

# --------------------
# Dataset
# --------------------
class BrainSeg2D(Dataset):
    """
    Returns:
      img: float tensor [1, H', W'] (z-scored)
      msk: long  tensor [H', W']   (class IDs 0..C-1)
    """
    def __init__(self, img_dir: Path, msk_dir: Path, size_hw=SIZE_HW, augment: bool=False):
        self.size_hw = size_hw
        self.augment = augment
        self.pairs = _pair_paths(img_dir, msk_dir)
        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs under:\n  {img_dir}\n  {msk_dir}")

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
        img = _resize_img(img, self.size_hw)
        img = _zscore(img)

        # Mask: grayscale -> contiguous indices
        msk = read_mask_grayscale_as_indices(msk_path, size_hw=self.size_hw)

        # Optional paired h-flip
        if self.augment and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            msk = torch.flip(msk, dims=[1])

        return img.contiguous(), msk.contiguous()

# --------------------
# Dataloader factory (anchored to this script)
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

# if __name__ == "__main__":
#     tr, va, te = create_dataloaders_here()
#     xb, yb = next(iter(tr))
#     print("X:", xb.shape, xb.dtype, "Y:", yb.shape, yb.dtype, "| uniques:", torch.unique(yb))
