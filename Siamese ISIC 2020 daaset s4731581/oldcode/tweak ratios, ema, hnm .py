import os, re, json, math, random, argparse, datetime, copy
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms as T
from PIL import Image, ImageFile
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------- AMP compatibility ---------------------
def has_torch_amp():
    return hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def amp_autocast_for(device):
    """Enable autocast only when actually using CUDA."""
    if device.type != "cuda":
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        return DummyCtx()
    if has_torch_amp():
        return torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        return torch.cuda.amp.autocast(enabled=True)

if has_torch_amp() and hasattr(torch.amp, "GradScaler"):
    GradScaler = torch.amp.GradScaler
else:
    GradScaler = torch.cuda.amp.GradScaler

# --------------------- Device selection & logging ---------------------
def choose_device(device_pref: str = "auto"):
    if device_pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warning] --device cuda requested but CUDA not available; using CPU.")
        return torch.device("cpu")
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warning] --device mps requested but MPS not available; using CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log_runtime(device):
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == "cuda":
        try:
            cc = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"CUDA (compiled for): {torch.version.cuda} | GPU: {name} | CC: {cc}")
        except Exception:
            pass

# --------------------- Data utils ---------------------
def find_image_dir(root: Path) -> Tuple[Optional[Path], int]:
    exts = {".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"}
    best, best_n = None, 0
    for p in root.rglob("*"):
        if p.is_dir():
            n = sum(1 for x in p.glob("*") if x.suffix in exts)
            if n > best_n:
                best, best_n = p, n
    return best, best_n

def pick_csv(root: Path) -> Optional[Path]:
    csvs = list(root.rglob("*.csv"))
    if not csvs: return None
    def score(p: Path):
        try:
            hdr = pd.read_csv(p, nrows=5, low_memory=False)
        except Exception:
            return (-1, p)
        cols = set(hdr.columns)
        label_like = any(c in cols for c in ["target","label","labels","benign_malignant","malignant","is_malignant"])
        id_like    = any(c in cols for c in ["isic_id","image_name","image","filename","file_name","id","name"])
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
            if m: by[m.group(1)].append(p)
    return by

def make_balanced_frame(df: pd.DataFrame, label_col: str = "target", seed: int = 42) -> pd.DataFrame:
    """Return a class-balanced subset of df by undersampling the majority class."""
    vc = df[label_col].value_counts()
    if 0 not in vc or 1 not in vc:
        raise SystemExit("Cannot build a balanced test set because one class is missing.")
    n = int(min(vc[0], vc[1]))
    df0 = df[df[label_col] == 0].sample(n=n, random_state=seed)
    df1 = df[df[label_col] == 1].sample(n=n, random_state=seed)
    return pd.concat([df0, df1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

# --------------------- Datasets ---------------------
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
    """Build pairs on the fly. y_pair=1 for same class (PP/NN), 0 for PN.
       Supports PN/PP/NN mix scheduling and hard negative mining."""
    def __init__(self, base_ds: ISIC2020Dataset, pn_pp_nn=(0.6,0.2,0.2), length=None):
        self.base = base_ds
        self.pn, self.pp, self.nn = pn_pp_nn
        self.length = length or len(base_ds)
        labels = self.base.df["target"].astype(int).tolist()
        self.pos_idx = [i for i, y in enumerate(labels) if y==1]
        self.neg_idx = [i for i, y in enumerate(labels) if y==0]
        assert self.pos_idx and self.neg_idx, "Both classes required."
        # HNM pool
        self.hard_negatives: Dict[int, List[int]] = {}
        self.hnm_frac: float = 0.0

    def set_mix(self, pn: float, pp: float, nn: float):
        s = pn + pp + nn
        self.pn, self.pp, self.nn = pn/s, pp/s, nn/s

    def set_hard_negatives(self, pool: Dict[int, List[int]], frac: float):
        self.hard_negatives = pool or {}
        self.hnm_frac = float(frac)

    def __len__(self): return self.length

    def __getitem__(self, _):
        r = random.random()
        if r < self.pn:
            # PN pair
            i = random.choice(self.pos_idx)
            use_hard = (self.hard_negatives and random.random() < self.hnm_frac and i in self.hard_negatives and len(self.hard_negatives[i])>0)
            if use_hard:
                j = random.choice(self.hard_negatives[i])
            else:
                j = random.choice(self.neg_idx)
            y = 0
        elif r < self.pn + self.pp:
            # PP pair
            i = random.choice(self.pos_idx); j = random.choice(self.pos_idx); y = 1
        else:
            # NN pair
            i = random.choice(self.neg_idx); j = random.choice(self.neg_idx); y = 1
        x1, _ = self.base[i]
        x2, _ = self.base[j]
        return x1, x2, y

# --------------------- Model (scratch) ---------------------
class SiameseResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None)   # train from scratch
        m.fc = nn.Identity()                 # 512-d
        self.encoder = m
        self.scale = nn.Parameter(torch.tensor(10.0))  # temperature for cosine logits
    def forward_once(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=1)
    def forward(self, x1, x2):
        z1 = self.forward_once(x1); z2 = self.forward_once(x2)
        logits = self.scale * F.cosine_similarity(z1, z2)  # (B,)
        return logits, z1, z2

# --------------------- EMA ---------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.995, device=None):
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(self.decay * v + (1.0 - self.decay) * msd[k].detach())

# --------------------- Embedding / Prototypes ---------------------
def maybe_subset(ds: Dataset, max_items: Optional[int]):
    if (max_items is None) or (len(ds) <= max_items): return ds
    idx = np.random.choice(len(ds), size=max_items, replace=False)
    return Subset(ds, idx.tolist())

@torch.no_grad()
def compute_embeddings(model, device, ds, batch=512, max_items=None):
    ds_sub = maybe_subset(ds, max_items)
    dl = DataLoader(ds_sub, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    all_z, all_y = [], []
    model.eval()
    for xb, yb in tqdm(dl, desc="Embed", leave=False):
        xb = xb.to(device, non_blocking=False)
        with amp_autocast_for(device):
            z = model.forward_once(xb)
        all_z.append(z.cpu()); all_y.append(yb)
    return torch.cat(all_z, 0), torch.cat(all_y, 0)

@torch.no_grad()
def prototype_eval(eval_model, device, train_ds, val_or_test_ds, batch=512, max_items=None):
    z_tr, y_tr = compute_embeddings(eval_model, device, train_ds, batch=batch, max_items=max_items)
    if (y_tr==0).any():
        c0 = F.normalize(z_tr[y_tr==0].mean(0, keepdim=True), dim=1)
    else:
        c0 = torch.zeros_like(z_tr[:1])
    if (y_tr==1).any():
        c1 = F.normalize(z_tr[y_tr==1].mean(0, keepdim=True), dim=1)
    else:
        c1 = torch.zeros_like(z_tr[:1])

    z_va, y_va = compute_embeddings(eval_model, device, val_or_test_ds, batch=batch, max_items=max_items)
    s1 = F.cosine_similarity(z_va, c1.expand_as(z_va))
    s0 = F.cosine_similarity(z_va, c0.expand_as(z_va))
    scores = (s1 - s0).cpu().numpy()
    y_true = y_va.numpy()
    y_pred = (scores > 0).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return acc, auc

# ---- Hard Negative Mining utilities ----
@torch.no_grad()
def _embeddings_for_indices(model, device, ds: ISIC2020Dataset, indices: np.ndarray, batch=512):
    sub = Subset(ds, indices.tolist())
    dl = DataLoader(sub, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    zs = []
    for xb, _ in tqdm(dl, desc="HNM embed", leave=False):
        xb = xb.to(device, non_blocking=False)
        with amp_autocast_for(device):
            z = model.forward_once(xb)
        zs.append(z.cpu())
    return torch.cat(zs, 0)  # (N, d), CPU

@torch.no_grad()
def build_hard_negative_pool(eval_model, device, ds: ISIC2020Dataset,
                             topk=5, max_pos=2000, max_neg=4000, batch=512, seed=42) -> Dict[int, List[int]]:
    """Return dict[pos_idx] = [neg_idx, ...] of 'hard' negatives (highest cosine)."""
    rng = np.random.default_rng(seed)
    labels = ds.df["target"].astype(int).values
    pos_idx = np.where(labels==1)[0]
    neg_idx = np.where(labels==0)[0]
    if len(pos_idx)==0 or len(neg_idx)==0:
        return {}

    pos_sel = rng.choice(pos_idx, size=min(len(pos_idx), max_pos), replace=False)
    neg_sel = rng.choice(neg_idx, size=min(len(neg_idx), max_neg), replace=False)

    z_pos = _embeddings_for_indices(eval_model, device, ds, pos_sel, batch=batch)  # (P, d)
    z_neg = _embeddings_for_indices(eval_model, device, ds, neg_sel, batch=batch)  # (N, d)

    # cosine since embeddings are normalized
    sim = z_pos @ z_neg.T  # (P, N)
    k = min(topk, z_neg.shape[0])
    vals, idxs = torch.topk(sim, k=k, dim=1, largest=True)
    pool = {}
    for row, pos_i in enumerate(pos_sel):
        hard_negs = neg_sel[idxs[row].cpu().numpy()].tolist()
        pool[int(pos_i)] = hard_negs
    return pool

# --------------------- Build datasets (with balanced test) ---------------------
def build_datasets(args, device):
    # 1) Resolve dataset root
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
    else:
        import kagglehub
        print("Downloading mirror with kagglehub ...")
        root = Path(kagglehub.dataset_download("ziadloo/isic-2020-256x256")).resolve()
    print("Dataset root:", root)

    # 2) Find images + CSV
    img_dir, n_imgs = find_image_dir(root)
    if not img_dir or n_imgs == 0:
        raise SystemExit("Could not locate image files.")
    csv_path = pick_csv(root)
    if not csv_path:
        raise SystemExit("Could not find a CSV with labels.")
    print(f"IMG_DIR: {img_dir} (files: {n_imgs})")
    print(f"CSV: {csv_path}")

    # 3) Read CSV, normalize id + label
    df = pd.read_csv(csv_path, low_memory=False)
    cand_id = ["isic_id","image_name","image","image_id","filename","file_name","id","name"]
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

    # 4) Map rows -> file paths
    by_stem = index_files_by_stem(img_dir)
    df["image_path"] = df["image_stem"].map(lambda s: str(by_stem[s][0]) if s in by_stem else None)
    missing = df["image_path"].isna().sum()
    print(f"Mapped images: {len(df)-missing}/{len(df)} (missing={missing})")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # 5) Split -> train / val / test (group-aware if patient_id exists)
    val_size = args.val_size
    test_size = args.test_size
    rnd = args.seed

    if "patient_id" in df.columns:
        print("Using patient-level, group-preserving split.")
        groups = df["patient_id"].astype(str)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rnd)
        idx = np.arange(len(df))
        trval_idx, test_idx = next(gss1.split(idx, groups=groups))
        df_trval = df.iloc[trval_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)

        val_prop = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=rnd)
        idx2 = np.arange(len(df_trval))
        tr_idx, va_idx = next(gss2.split(idx2, groups=df_trval["patient_id"].astype(str)))

        train_df = df_trval.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_trval.iloc[va_idx].reset_index(drop=True)
    else:
        df_trval, df_test = train_test_split(
            df, test_size=test_size, stratify=df["target"].astype(int), random_state=rnd
        )
        val_prop = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            df_trval, test_size=val_prop, stratify=df_trval["target"].astype(int), random_state=rnd
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)

    # Balanced test set by undersampling majority class
    test_bal_df = make_balanced_frame(df_test, label_col="target", seed=rnd)

    # 6) Datasets / transforms
    img_size = args.img_size
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    eval_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(), T.Normalize(mean, std),
    ])

    ds_train    = ISIC2020Dataset(train_df,     train_tfms)
    ds_val      = ISIC2020Dataset(val_df,       eval_tfms)
    ds_test_bal = ISIC2020Dataset(test_bal_df,  eval_tfms)

    print(f"Datasets ready ✓  train: {len(ds_train)}  val: {len(ds_val)}  test: {len(df_test)}  test_bal: {len(ds_test_bal)}")
    print("Class balance →",
          f"train: {train_df['target'].mean():.3f} pos",
          f"val: {val_df['target'].mean():.3f} pos",
          f"test: {df_test['target'].mean():.3f} pos",
          f"test_bal: {test_bal_df['target'].mean():.3f} pos")

    return ds_train, ds_val, ds_test_bal

# --------------------- Train ---------------------
def adjust_lr_on_drop(optimizer, factor=0.5, min_lr=1e-6):
    for pg in optimizer.param_groups:
        old = pg.get("lr", 0.0)
        new = max(old * factor, min_lr)
        pg["lr"] = new
    return optimizer.param_groups[0]["lr"]

def maybe_apply_mix_schedule(pair_ds: PairDataset, epoch: int, args):
    if not args.use_mix_schedule:
        # static mix from args
        pair_ds.set_mix(*tuple(args.pn_pp_nn))
        return
    # Scheduled mixes
    if epoch <= args.mix_ep1:
        mix = (0.80, 0.10, 0.10)
    elif epoch <= args.mix_ep2:
        mix = (0.70, 0.15, 0.15)
    else:
        mix = (0.60, 0.20, 0.20)
    pair_ds.set_mix(*mix)

def train(args):
    # Repro & device
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = choose_device(args.device)
    log_runtime(device)

    # I/O
    run_dir = Path(args.out_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,val_pair_loss,val_proto_acc,val_proto_auc\n")

    # Data
    ds_train, ds_val, ds_test_bal = build_datasets(args, device)

    # Pair loaders — Windows-safe defaults (workers=0)
    pair_ds_train = PairDataset(ds_train, pn_pp_nn=tuple(args.pn_pp_nn), length=len(ds_train))
    pair_ds_val   = PairDataset(ds_val,   pn_pp_nn=(0.5,0.25,0.25),     length=len(ds_val))
    # set initial mix (may be overridden per-epoch by schedule)
    maybe_apply_mix_schedule(pair_ds_train, 1, args)

    pair_dl_train = DataLoader(pair_ds_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=(device.type=="cuda"))
    pair_dl_val   = DataLoader(pair_ds_val,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=(device.type=="cuda"))
    print("Pair loaders:", len(pair_dl_train), len(pair_dl_val))

    # Model / Opt / Loss / EMA
    model = SiameseResNet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device.type=="cuda"))
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    best_acc = -1.0
    best_test_acc = -1.0
    last_test_acc = None
    stopped_early = False

    for epoch in range(1, args.epochs+1):
        # ----- PN/PP/NN schedule -----
        maybe_apply_mix_schedule(pair_ds_train, epoch, args)

        # -------- Train --------
        model.train()
        running, n_batches = 0.0, 0
        pbar = tqdm(pair_dl_train, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for x1, x2, y in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = y.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast_for(device):
                logits, _, _ = model(x1, x2)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            running += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, n_batches)

        # -------- Val (pair loss) --------
        model.eval()
        val_running, val_nb = 0.0, 0
        pbar_v = tqdm(pair_dl_val, desc=f"Epoch {epoch}/{args.epochs} [val pairs]", leave=False)
        with torch.no_grad():
            for x1, x2, y in pbar_v:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                y  = y.float().to(device, non_blocking=True)
                with amp_autocast_for(device):
                    logits, _, _ = model(x1, x2)
                    vloss = criterion(logits, y)
                val_running += vloss.item(); val_nb += 1
                pbar_v.set_postfix(loss=f"{vloss.item():.4f}")
        val_pair_loss = val_running / max(1, val_nb)

        # -------- Prototype eval (optional for speed) --------
        eval_model = ema.ema if ema is not None else model
        if args.embed_every > 0 and (epoch % args.embed_every == 0):
            val_proto_acc, val_proto_auc = prototype_eval(
                eval_model, device, ds_train, ds_val,
                batch=args.embed_batch, max_items=args.embed_max
            )
        else:
            val_proto_acc, val_proto_auc = float("nan"), float("nan")

        # -------- Log --------
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_pair_loss:.6f},{val_proto_acc:.6f},{val_proto_auc:.6f}\n")

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | val_pair_loss={val_pair_loss:.4f} | "
              f"proto_acc={val_proto_acc:.4f} | proto_auc={val_proto_auc:.4f}")

        # -------- Save every N epochs --------
        if epoch % args.save_every == 0:
            ckpt_path = run_dir / f"siamese_resnet18_epoch{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": (ema.ema.state_dict() if ema is not None else None),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(args)
            }, ckpt_path)
            print("Saved:", ckpt_path)

        # Track best by proto-acc (if computed)
        if not math.isnan(val_proto_acc) and (val_proto_acc > best_acc):
            best_acc = val_proto_acc
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_state": (ema.ema.state_dict() if ema is not None else None)},
                       run_dir / "best.pt")

        # -------- Periodic TEST on balanced set --------
        if epoch % args.test_every == 0:
            test_bal_acc, test_bal_auc = prototype_eval(
                eval_model, device, ds_train, ds_test_bal,
                batch=args.embed_batch, max_items=args.embed_max
            )
            with open(metrics_path, "a") as f:
                f.write(f"test_bal@{epoch},,,{test_bal_acc:.6f},{test_bal_auc:.6f}\n")
            print(f"[TEST-BAL@{epoch}] proto_acc={test_bal_acc:.4f} | proto_auc={test_bal_auc:.4f}")

            # Save best-test model
            if test_bal_acc > best_test_acc:
                best_test_acc = test_bal_acc
                torch.save({"epoch": epoch,
                            "model_state": model.state_dict(),
                            "ema_state": (ema.ema.state_dict() if ema is not None else None)},
                           run_dir / "best_test.pt")

            # Early stop if threshold reached
            if test_bal_acc >= args.early_stop_acc:
                es_path = run_dir / f"early_stop_ep{epoch:02d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "ema_state": (ema.ema.state_dict() if ema is not None else None),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(args),
                    "test_bal_acc": test_bal_acc,
                    "test_bal_auc": test_bal_auc
                }, es_path)
                print(f"[EARLY-STOP] Test acc {test_bal_acc:.4f} ≥ {args.early_stop_acc:.2f}. Saved {es_path}")
                stopped_early = True
                break

            # Optional LR decay on test accuracy drop
            if args.lr_decay_on_drop and (last_test_acc is not None):
                if test_bal_acc < (last_test_acc - args.drop_tolerance):
                    new_lr = adjust_lr_on_drop(optimizer, factor=args.lr_decay_factor, min_lr=args.lr_min)
                    print(f"[LR-DECAY] test acc dropped {last_test_acc:.4f} → {test_bal_acc:.4f}. New lr={new_lr:.2e}")
            last_test_acc = test_bal_acc

        # -------- HNM refresh --------
        if args.hnm and (epoch >= args.hnm_start) and (epoch % args.hnm_every == 0):
            print("[HNM] Building hard negative pool ...")
            hnm_model = ema.ema if ema is not None else model
            pool = build_hard_negative_pool(
                hnm_model, device, ds_train,
                topk=args.hnm_topk, max_pos=args.hnm_max_pos, max_neg=args.hnm_max_neg,
                batch=args.hnm_embed_batch, seed=args.seed
            )
            pair_ds_train.set_hard_negatives(pool, frac=args.hnm_frac)
            print(f"[HNM] Pool built for {len(pool)} positives; using frac={args.hnm_frac:.2f}")

    # ----- Final: evaluate on BALANCED test set if not early-stopped -----
    if not stopped_early:
        eval_model = ema.ema if ema is not None else model
        test_bal_acc, test_bal_auc = prototype_eval(
            eval_model, device, ds_train, ds_test_bal,
            batch=args.embed_batch, max_items=args.embed_max
        )
        with open(metrics_path, "a") as f:
            f.write(f"test_bal@final,,,{test_bal_acc:.6f},{test_bal_auc:.6f}\n")
        print(f"[TEST-BAL@final] proto_acc={test_bal_acc:.4f} | proto_auc={test_bal_auc:.4f}")

    # Save config & last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema.state_dict() if ema is not None else None)
    }, run_dir / "last.pt")
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Run dir:", run_dir)
    print("Metrics:", metrics_path)

# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser("Siamese ISIC2020 (Windows) — scratch + schedule + EMA + HNM")
    p.add_argument("--data-root", type=str, default="",
                   help="Folder of mirror (optional). If empty, downloads via kagglehub.")
    p.add_argument("--out-dir", type=str, default="./runs", help="Where to save logs/ckpts.")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=256, help="256 (mirror) or larger like 320/384.")
    p.add_argument("--pn-pp-nn", type=float, nargs=3, default=(0.75,0.125,0.125),
                   help="Default pair mix when schedule disabled: PN, PP, NN")
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--embed-every", type=int, default=5,
                   help="Prototype eval every N epochs (0/neg to disable).")
    p.add_argument("--embed-batch", type=int, default=512)
    p.add_argument("--embed-max", type=int, default=20000,
                   help="Max samples for embedding (None = all).")
    p.add_argument("--workers", type=int, default=0,
                   help="Dataloader workers (Windows-safe default=0).")
    p.add_argument("--val-size",  type=float, default=0.15,
                   help="Validation share of the whole dataset.")
    p.add_argument("--test-size", type=float, default=0.15,
                   help="Test share of the whole dataset.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto","cuda","cpu","mps"],
                   help="Force device. 'auto' picks CUDA if available, else MPS, else CPU.")

    # Testing cadence, early stop, LR response to drops
    p.add_argument("--test-every", type=int, default=5, help="Test cadence (epochs).")
    p.add_argument("--early-stop-acc", type=float, default=0.80,
                   help="Early stop on test acc ≥ this.")
    p.add_argument("--lr-decay-on-drop", action="store_true",
                   help="Reduce LR if test acc falls.")
    p.add_argument("--drop-tolerance", type=float, default=0.003,
                   help="Min absolute drop to trigger LR decay.")
    p.add_argument("--lr-decay-factor", type=float, default=0.5,
                   help="Multiply LR by this on drop.")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Lower bound for LR.")

    # Mix schedule (optional)
    p.add_argument("--use-mix-schedule", action="store_true",
                   help="Enable PN/PP/NN schedule: (1-ep1) 0.80/0.10/0.10, (ep1-ep2) 0.70/0.15/0.15, (ep2+) 0.60/0.20/0.20")
    p.add_argument("--mix-ep1", type=int, default=5, help="End epoch of phase 1.")
    p.add_argument("--mix-ep2", type=int, default=20, help="End epoch of phase 2.")

    # EMA
    p.add_argument("--ema", action="store_true", help="Use EMA weights for eval and HNM.")
    p.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay (0..1).")

    # HNM
    p.add_argument("--hnm", action="store_true", help="Enable hard negative mining.")
    p.add_argument("--hnm-start", type=int, default=10, help="First epoch to build HNM pool.")
    p.add_argument("--hnm-every", type=int, default=5, help="Rebuild HNM pool every N epochs.")
    p.add_argument("--hnm-frac", type=float, default=0.5, help="Fraction of PN pairs drawn from HNM pool.")
    p.add_argument("--hnm-topk", type=int, default=5, help="Top-k hardest negatives per positive.")
    p.add_argument("--hnm-max-pos", type=int, default=2000, help="Max positives to embed for HNM.")
    p.add_argument("--hnm-max-neg", type=int, default=4000, help="Max negatives to embed for HNM.")
    p.add_argument("--hnm-embed-batch", type=int, default=512, help="Batch size for HNM embeddings.")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
