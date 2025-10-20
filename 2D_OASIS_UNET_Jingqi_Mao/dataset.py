from pathlib import Path
import os

def _collect_all(folder: Path):
    out = []
    for pat in ("*.nii.png", "*.png"):
        out.extend(folder.glob(pat))
    return sorted(out)

def _to_seg_stem(img_stem: str) -> str:
    # case_001_slice_10  -> seg_001_slice_10
    return "seg_" + img_stem[len("case_"):] if img_stem.startswith("case_") else img_stem

def _pair_paths(img_dir: Path, msk_dir: Path):
    imgs = _collect_all(img_dir)
    msks = _collect_all(msk_dir)

    # build mask lookup by stem, tolerant of double suffix (…nii.png)
    def clean_stem(p: Path) -> str:
        s = p.stem               # removes last suffix (.png)
        return Path(s).stem if s.endswith(".nii") else s  # remove .nii if present
    msk_by_stem = {clean_stem(Path(p)): str(p) for p in msks}

    pairs, missing = [], []
    for ip in imgs:
        ist = clean_stem(Path(ip))
        tgt = _to_seg_stem(ist)
        mp = msk_by_stem.get(tgt)
        if mp and os.path.exists(mp):
            pairs.append((str(ip), mp))
        else:
            missing.append(f"{Path(ip).name} -> {tgt}[.png|.nii.png]")

    if not pairs:
        print("\n--- Pairing diagnostics ---")
        print("Sample images:", [Path(p).name for p in imgs[:5]])
        print("Sample masks :", [Path(p).name for p in msks[:5]])
        print("First missing:", missing[:10])
        print("---------------------------\n")

    pairs.sort(key=lambda t: os.path.basename(t[0]))
    return pairs

def main():
    img_dir = Path(__file__).parent/"keras_png_slices_data/keras_png_slices_train"
    msk_dir = Path(__file__).parent/"keras_png_slices_data/keras_png_slices_seg_train"
    pairs = _pair_paths(img_dir, msk_dir)
    print("Found pairs:", len(pairs))
    print("First 3:", [ (Path(i).name, Path(m).name) for i,m in pairs[:3] ])
if __name__ == "__main__":
    main()