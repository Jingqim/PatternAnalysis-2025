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

# --------------------- AMP helpers ---------------------
def has_torch_amp():
    return hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def amp_autocast_for(device):
    if device.type != "cuda":
        class Dummy:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        return Dummy()
    if has_torch_amp():
        return torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        return torch.cuda.amp.autocast(enabled=True)

GradScaler = (torch.amp.GradScaler if has_torch_amp() and hasattr(torch.amp,"GradScaler")
              else torch.cuda.amp.GradScaler)

# --------------------- Device & logging ---------------------
def choose_device(pref="auto"):
    if pref=="cuda":
        if torch.cuda.is_available(): return torch.device("cuda")
        print("[warning] CUDA requested but not available; falling back to CPU"); return torch.device("cpu")
    if pref=="cpu": return torch.device("cpu")
    if pref=="mps":
        if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warning] MPS requested but not available; CPU"); return torch.device("cpu")
    # auto
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def log_runtime(device):
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type=="cuda":
        try:
            print(f"CUDA (compiled): {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)} | CC: {torch.cuda.get_device_capability(0)}")
        except Exception: pass

# --------------------- Data utils ---------------------
def find_image_dir(root: Path) -> Tuple[Optional[Path], int]:
    exts = {".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"}
    best, best_n = None, 0
    for p in root.rglob("*"):
        if p.is_dir():
            n = sum(1 for x in p.glob("*") if x.suffix in exts)
            if n > best_n: best, best_n = p, n
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
        id_like    = any(c in cols for c in ["isic_id","image_name","image","image_id","filename","file_name","id","name"])
        return (int(label_like)+int(id_like), p)
    scored = sorted((score(p) for p in csvs), reverse=True)
    return scored[0][1] if scored and scored[0][0]>0 else csvs[0]

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

def make_balanced_frame(df: pd.DataFrame, label_col="target", seed=42) -> pd.DataFrame:
    vc = df[label_col].value_counts()
    if 0 not in vc or 1 not in vc:
        raise SystemExit("Balanced test requires both classes.")
    n = int(min(vc[0], vc[1]))
    df0 = df[df[label_col]==0].sample(n=n, random_state=seed)
    df1 = df[df[label_col]==1].sample(n=n, random_state=seed)
    return pd.concat([df0,df1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

# --------- Metadata encoding ---------
def build_meta(df: pd.DataFrame):
    cols_present = df.columns
    age = df["age_approx"].astype(float).fillna(df["age_approx"].median())/100.0 if "age_approx" in cols_present else pd.Series([0.0]*len(df))
    sex = df["sex"].astype(str).str.lower() if "sex" in cols_present else pd.Series(["unknown"]*len(df))
    site = df["anatom_site_general_challenge"].astype(str).str.lower() if "anatom_site_general_challenge" in cols_present else pd.Series(["unknown"]*len(df))

    sex_map = {"male":0, "female":1}
    sex_idx = sex.map(lambda s: sex_map.get(s,2))
    sex_oh = np.eye(3)[sex_idx.clip(0,2).values]   # male,female,unknown

    top_sites = ["lower extremity","upper extremity","torso","head/neck","palms/soles","oral/genital","back","chest","abdomen","unknown"]
    site_map = {k:i for i,k in enumerate(top_sites)}
    site_vals = site.fillna("unknown")
    site_idx = site_vals.map(lambda s: site_map.get(s, len(top_sites)-1))
    site_oh = np.eye(len(top_sites))[site_idx.clip(0,len(top_sites)-1).values]

    meta = np.concatenate([age.values.reshape(-1,1), sex_oh, site_oh], axis=1).astype(np.float32)
    names = (["age_norm"] + [f"sex_{k}" for k in ["male","female","unknown"]] +
             [f"site_{k.replace('/','_').replace(' ','_')}" for k in top_sites])
    return meta, names

# --------------------- Datasets ---------------------
class ISIC2020Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transforms=None, meta_vec: Optional[np.ndarray]=None):
        self.df = frame.reset_index(drop=True)
        self.t = transforms
        if meta_vec is None:
            self.meta = np.zeros((len(self.df),0),dtype=np.float32)
        else:
            self.meta = meta_vec.astype(np.float32)
        self.meta_dim = self.meta.shape[1]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        if self.t: img = self.t(img)
        y = int(r["target"])
        meta = torch.from_numpy(self.meta[i]) if self.meta_dim>0 else torch.empty(0)
        return img, y, meta

class PairDataset(Dataset):
    """Pairs for BCE/contrastive, with optional hard-negative pool."""
    def __init__(self, base: ISIC2020Dataset, pn_pp_nn=(0.6,0.2,0.2), length=None):
        self.base = base
        self.pn, self.pp, self.nn = pn_pp_nn
        self.length = length or len(base)
        labels = self.base.df["target"].astype(int).tolist()
        self.pos_idx = [i for i,y in enumerate(labels) if y==1]
        self.neg_idx = [i for i,y in enumerate(labels) if y==0]
        assert self.pos_idx and self.neg_idx, "Both classes required."
        self.hard_negatives: Dict[int, List[int]] = {}
        self.hnm_frac = 0.0
    def set_mix(self, pn, pp, nn):
        s = pn+pp+nn; self.pn, self.pp, self.nn = pn/s, pp/s, nn/s
    def set_hard_negatives(self, pool, frac: float):
        self.hard_negatives = pool or {}; self.hnm_frac = float(frac)
    def __len__(self): return self.length
    def __getitem__(self, _):
        r = random.random()
        if r < self.pn:
            i = random.choice(self.pos_idx)
            use_hard = (self.hard_negatives and random.random()<self.hnm_frac and i in self.hard_negatives and self.hard_negatives[i])
            j = random.choice(self.hard_negatives[i]) if use_hard else random.choice(self.neg_idx)
            y_pair = 0
        elif r < self.pn+self.pp:
            i = random.choice(self.pos_idx); j = random.choice(self.pos_idx); y_pair = 1
        else:
            i = random.choice(self.neg_idx); j = random.choice(self.neg_idx); y_pair = 1
        x1,y1,m1 = self.base[i]
        x2,y2,m2 = self.base[j]
        return x1,x2,y_pair,m1,m2

# --------------------- Model: mask + meta fusion + ResNet18 ---------------------
class SiameseResNet18(nn.Module):
    def __init__(self, mask_mode="off", meta_dim=0, mask_l1=0.0):
        super().__init__()
        self.mask_mode = mask_mode
        self.mask_l1 = mask_l1
        in_ch = 3 + (1 if mask_mode!="off" else 0)

        # mask head (learned attention producing 1-ch map in [0,1])
        self.mask_head = None
        if mask_mode == "learn":
            self.mask_head = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1), nn.Sigmoid()
            )

        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Identity()
        self.encoder = m  # 512-d

        self.meta_dim = int(meta_dim)
        if self.meta_dim>0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(self.meta_dim, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 64), nn.ReLU(inplace=True)
            )
            self.fuse = nn.Linear(512+64, 512)
        else:
            self.meta_mlp = None
            self.fuse = None

        self.scale = nn.Parameter(torch.tensor(10.0))  # for BCE/contrastive logits

    def _augment_with_mask(self, x):
        mask = None
        if self.mask_head is not None:
            mask = self.mask_head(x)                          # [B,1,H,W]
            x = torch.cat([x, mask], dim=1)
        return x, mask

    def forward_once(self, x, meta: Optional[torch.Tensor]=None):
        x_aug, mask = self._augment_with_mask(x)
        z = self.encoder(x_aug)                               # [B,512]
        if (meta is not None) and (self.meta_dim>0) and meta.numel()>0:
            if meta.dim()==1: meta = meta.unsqueeze(0)
            m = self.meta_mlp(meta)
            z = torch.cat([z, m], dim=1)
            z = self.fuse(z)
        z = F.normalize(z, dim=1)
        return z, mask

    def pair_forward(self, x1, x2, meta1=None, meta2=None):
        z1, mask1 = self.forward_once(x1, meta1)
        z2, mask2 = self.forward_once(x2, meta2)
        logits = self.scale * F.cosine_similarity(z1, z2)     # (B,)
        masks = []
        if mask1 is not None: masks.append(mask1)
        if mask2 is not None: masks.append(mask2)
        return logits, z1, z2, masks

# --------------------- EMA ---------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.995, device=None):
        self.ema = copy.deepcopy(model).eval()
        if device is not None: self.ema.to(device)
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = float(decay)
    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(self.decay*v + (1.0-self.decay)*msd[k].detach())

# --------------------- Embedding / Prototypes ---------------------
def maybe_subset(ds: Dataset, max_items: Optional[int]):
    if (max_items is None) or (len(ds) <= max_items): return ds
    idx = np.random.choice(len(ds), size=max_items, replace=False)
    return Subset(ds, idx.tolist())

@torch.no_grad()
def compute_embeddings(model, device, ds, batch=512, max_items=None):
    sub = maybe_subset(ds, max_items)
    dl = DataLoader(sub, batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    zs, ys = [], []
    model.eval()
    for xb, yb, mb in tqdm(dl, desc="Embed", leave=False):
        xb = xb.to(device, non_blocking=False)
        mb = mb.to(device, non_blocking=False) if mb.numel()>0 else None
        with amp_autocast_for(device):
            z, _ = model.forward_once(xb, mb)
        zs.append(z.cpu()); ys.append(yb)
    return torch.cat(zs,0), torch.cat(ys,0)

@torch.no_grad()
def prototype_eval(eval_model, device, train_ds, val_or_test_ds, batch=512, max_items=None):
    z_tr, y_tr = compute_embeddings(eval_model, device, train_ds, batch=batch, max_items=max_items)
    c0 = F.normalize(z_tr[y_tr==0].mean(0, keepdim=True), dim=1) if (y_tr==0).any() else torch.zeros_like(z_tr[:1])
    c1 = F.normalize(z_tr[y_tr==1].mean(0, keepdim=True), dim=1) if (y_tr==1).any() else torch.zeros_like(z_tr[:1])
    z_va, y_va = compute_embeddings(eval_model, device, val_or_test_ds, batch=batch, max_items=max_items)
    s1 = F.cosine_similarity(z_va, c1.expand_as(z_va))
    s0 = F.cosine_similarity(z_va, c0.expand_as(z_va))
    scores = (s1 - s0).cpu().numpy(); y_true = y_va.numpy()
    y_pred = (scores > 0).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, scores)
    except ValueError: auc = float("nan")
    return acc, auc

# --------------------- HNM for pair training (kept) ---------------------
@torch.no_grad()
def _embed_indices(model, device, ds: ISIC2020Dataset, indices: np.ndarray, batch=512):
    sub = Subset(ds, indices.tolist())
    dl = DataLoader(sub, batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    zs = []
    for xb, _, mb in tqdm(dl, desc="HNM embed", leave=False):
        xb = xb.to(device, non_blocking=False)
        mb = mb.to(device, non_blocking=False) if mb.numel()>0 else None
        with amp_autocast_for(device):
            z, _ = model.forward_once(xb, mb)
        zs.append(z.cpu())
    return torch.cat(zs,0)

@torch.no_grad()
def build_hard_negative_pool(eval_model, device, ds: ISIC2020Dataset,
                             topk=10, max_pos=2000, max_neg=4000, batch=512, seed=42) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    labels = ds.df["target"].astype(int).values
    pos_idx = np.where(labels==1)[0]; neg_idx = np.where(labels==0)[0]
    if len(pos_idx)==0 or len(neg_idx)==0: return {}
    pos_sel = rng.choice(pos_idx, size=min(len(pos_idx), max_pos), replace=False)
    neg_sel = rng.choice(neg_idx, size=min(len(neg_idx), max_neg), replace=False)
    z_pos = _embed_indices(eval_model, device, ds, pos_sel, batch=batch)
    z_neg = _embed_indices(eval_model, device, ds, neg_sel, batch=batch)
    sim = z_pos @ z_neg.T
    k = min(topk, z_neg.shape[0]); _, idxs = torch.topk(sim, k=k, dim=1)
    pool = {int(pos_sel[row]): neg_sel[idxs[row].cpu().numpy()].tolist() for row in range(len(pos_sel))}
    return pool

# --------------------- Losses ---------------------
def contrastive_loss_from_embeddings(z1, z2, y, margin=0.5):
    cos = F.cosine_similarity(z1, z2)
    dist = torch.clamp(2 - 2*cos, min=0).sqrt()
    pos = y * (dist**2)
    neg = (1-y) * (torch.clamp(margin - dist, min=0)**2)
    return (pos + neg).mean()

def mine_inbatch_triplets(z, y, margin=0.2):
    B = z.size(0)
    with torch.no_grad():
        sim = z @ z.t()
        y_mat = y.unsqueeze(1)==y.unsqueeze(0)
        pos_mask = y_mat & ~torch.eye(B, dtype=torch.bool, device=y.device)
        neg_mask = ~y_mat
        pos_sim = sim.clone(); pos_sim[~pos_mask] = -1e9
        pos_idx = torch.argmax(pos_sim, dim=1)
        neg_sim = sim.clone(); neg_sim[~neg_mask] = -1e9
        neg_idx = torch.argmax(neg_sim, dim=1)
    anchor = torch.arange(B, device=z.device)
    return anchor, pos_idx, neg_idx

def triplet_loss_inbatch(z, y, margin=0.2):
    a, p, n = mine_inbatch_triplets(z, y, margin)
    za, zp, zn = z[a], z[p], z[n]
    def d(u,v): return torch.clamp(2 - 2*(u*v).sum(dim=1), min=0).sqrt()
    dap = d(za,zp); dan = d(za,zn)
    loss = torch.clamp(dap - dan + margin, min=0).mean()
    return loss

# --------------------- Build datasets (with metadata alignment fix) ---------------------
def build_datasets(args, device):
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
    else:
        import kagglehub
        print("Downloading mirror with kagglehub ...")
        root = Path(kagglehub.dataset_download("ziadloo/isic-2020-256x256")).resolve()
    print("Dataset root:", root)

    img_dir, n_imgs = find_image_dir(root)
    if not img_dir or n_imgs==0: raise SystemExit("Could not locate image files.")
    csv_path = pick_csv(root)
    if not csv_path: raise SystemExit("Could not find a CSV with labels.")
    print(f"IMG_DIR: {img_dir} (files: {n_imgs})")
    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    cand_id = ["isic_id","image_name","image","image_id","filename","file_name","id","name"]
    id_col = next((c for c in cand_id if c in df.columns), None)
    if id_col is None:
        for c in df.columns:
            sample = df[c].astype(str).head(50).tolist()
            if sum(bool(re.search(r"ISIC_\d+", s)) for s in sample) >= 10:
                id_col = c; break
    if id_col is None: raise SystemExit("Could not identify image id column.")

    if "target" in df.columns:
        df["target"] = df["target"].astype(int)
    elif "benign_malignant" in df.columns:
        df["target"] = (df["benign_malignant"].astype(str).str.lower()=="malignant").astype(int)
    elif "label" in df.columns:
        df["target"] = df["label"].astype(int)
    else:
        raise SystemExit("No label column found (target/benign_malignant/label).")

    df["image_stem"] = df[id_col].astype(str).map(to_isic_stem)
    by_stem = index_files_by_stem(img_dir)
    df["image_path"] = df["image_stem"].map(lambda s: str(by_stem[s][0]) if s in by_stem else None)
    missing = df["image_path"].isna().sum()
    print(f"Mapped images: {len(df)-missing}/{len(df)} (missing={missing})")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # ---- IMPORTANT: freeze original row indices for metadata alignment
    df["orig_idx"] = np.arange(len(df))

    # splits (group-aware if patient_id present)
    rnd = args.seed
    if "patient_id" in df.columns:
        print("Using patient-level, group-preserving split.")
        groups = df["patient_id"].astype(str)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=rnd)
        idx = np.arange(len(df)); trval_idx, test_idx = next(gss1.split(idx, groups=groups))
        df_trval = df.iloc[trval_idx].reset_index(drop=True); df_test = df.iloc[test_idx].reset_index(drop=True)
        val_prop = args.val_size/(1.0-args.test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=rnd)
        idx2 = np.arange(len(df_trval)); tr_idx, va_idx = next(gss2.split(idx2, groups=df_trval["patient_id"].astype(str)))
        train_df = df_trval.iloc[tr_idx].reset_index(drop=True); val_df = df_trval.iloc[va_idx].reset_index(drop=True)
    else:
        df_trval, df_test = train_test_split(df, test_size=args.test_size, stratify=df["target"].astype(int), random_state=rnd)
        val_prop = args.val_size/(1.0-args.test_size)
        train_df, val_df = train_test_split(df_trval, test_size=val_prop, stratify=df_trval["target"].astype(int), random_state=rnd)
        train_df = train_df.reset_index(drop=True); val_df = val_df.reset_index(drop=True); df_test = df_test.reset_index(drop=True)

    test_bal_df = make_balanced_frame(df_test, label_col="target", seed=rnd)

    # transforms
    img_size = args.img_size
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    train_tfms = T.Compose([ T.Resize((img_size,img_size)),
                             T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                             T.ToTensor(), T.Normalize(mean,std) ])
    eval_tfms  = T.Compose([ T.Resize((img_size,img_size)),
                             T.ToTensor(), T.Normalize(mean,std) ])

    # metadata (ALIGNED BY orig_idx)
    if args.use_metadata:
        meta_all, meta_names = build_meta(df)
        meta_train = meta_all[train_df["orig_idx"].to_numpy()]
        meta_val   = meta_all[val_df["orig_idx"].to_numpy()]
        meta_testb = meta_all[test_bal_df["orig_idx"].to_numpy()]
        meta_dim = meta_all.shape[1]
        print(f"[META] enabled. dim={meta_dim}. Example cols: {meta_names[:6]} ...")
    else:
        meta_train = meta_val = meta_testb = None
        meta_dim = 0
        print("[META] disabled.")

    ds_train    = ISIC2020Dataset(train_df,     train_tfms, meta_train)
    ds_val      = ISIC2020Dataset(val_df,       eval_tfms,  meta_val)
    ds_test_bal = ISIC2020Dataset(test_bal_df,  eval_tfms,  meta_testb)

    print(f"Datasets ✓  train:{len(ds_train)}  val:{len(ds_val)}  test:{len(df_test)}  test_bal:{len(ds_test_bal)}")
    print("Class balance →",
          f"train: {train_df['target'].mean():.3f} pos",
          f"val: {val_df['target'].mean():.3f} pos",
          f"test: {df_test['target'].mean():.3f} pos",
          f"test_bal: {test_bal_df['target'].mean():.3f} pos")

    return ds_train, ds_val, ds_test_bal, meta_dim

# --------------------- Training ---------------------
def adjust_lr_on_drop(optim, factor=0.5, min_lr=1e-6):
    for pg in optim.param_groups:
        pg["lr"] = max(pg.get("lr",0.0)*factor, min_lr)
    return optim.param_groups[0]["lr"]

def maybe_apply_mix_schedule(pair_ds: PairDataset, epoch: int, args):
    if not args.use_mix_schedule:
        pair_ds.set_mix(*tuple(args.pn_pp_nn)); return
    mix = (0.80,0.10,0.10) if epoch<=args.mix_ep1 else (0.70,0.15,0.15) if epoch<=args.mix_ep2 else (0.60,0.20,0.20)
    pair_ds.set_mix(*mix)

def train(args):
    # Repro & device
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = choose_device(args.device); log_runtime(device)

    # I/O
    run_dir = Path(args.out_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,val_pair_loss,val_proto_acc,val_proto_auc\n")

    # Data
    ds_train, ds_val, ds_test_bal, meta_dim = build_datasets(args, device)

    # Model
    model = SiameseResNet18(mask_mode=args.mask_mode, meta_dim=meta_dim, mask_l1=args.mask_l1).to(device)
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    # Pair loaders for BCE/contrastive
    pair_ds_train = PairDataset(ds_train, pn_pp_nn=tuple(args.pn_pp_nn), length=len(ds_train))
    pair_ds_val   = PairDataset(ds_val,   pn_pp_nn=(0.5,0.25,0.25), length=len(ds_val))
    maybe_apply_mix_schedule(pair_ds_train, 1, args)
    pair_dl_train = DataLoader(pair_ds_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=(device.type=="cuda"))
    pair_dl_val   = DataLoader(pair_ds_val,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # Image loaders for triplet (in-batch mining)
    img_dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type=="cuda"))
    img_dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type=="cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device.type=="cuda"))
    criterion_bce = nn.BCEWithLogitsLoss()

    best_acc = -1.0
    best_test_acc = -1.0
    last_test_acc = None
    stopped_early = False

    for epoch in range(1, args.epochs+1):
        model.train()
        running, n_batches = 0.0, 0

        if args.loss == "triplet":
            pbar = tqdm(img_dl_train, desc=f"Epoch {epoch}/{args.epochs} [triplet]", leave=False)
            for xb, yb, mb in pbar:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True) if mb.numel()>0 else None

                optimizer.zero_grad(set_to_none=True)
                with amp_autocast_for(device):
                    z, mask = model.forward_once(xb, mb)
                    loss = triplet_loss_inbatch(z, yb, margin=args.triplet_margin)
                    if mask is not None and args.mask_l1>0:
                        loss = loss + args.mask_l1 * mask.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                if ema is not None: ema.update(model)

                running += loss.item(); n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        else:
            maybe_apply_mix_schedule(pair_ds_train, epoch, args)
            pbar = tqdm(pair_dl_train, desc=f"Epoch {epoch}/{args.epochs} [pairs-{args.loss}]", leave=False)
            for x1,x2,y_pair,m1,m2 in pbar:
                x1=x1.to(device,non_blocking=True); x2=x2.to(device,non_blocking=True)
                y_pair=y_pair.float().to(device,non_blocking=True)
                m1 = m1.to(device,non_blocking=True) if m1.numel()>0 else None
                m2 = m2.to(device,non_blocking=True) if m2.numel()>0 else None

                optimizer.zero_grad(set_to_none=True)
                with amp_autocast_for(device):
                    logits, z1, z2, masks = model.pair_forward(x1,x2,m1,m2)
                    if args.loss=="bce":
                        loss = criterion_bce(logits, y_pair)
                    else:
                        loss = contrastive_loss_from_embeddings(z1, z2, y_pair, margin=args.contrastive_margin)
                    if masks and args.mask_l1>0:
                        loss = loss + args.mask_l1 * torch.stack([m.mean() for m in masks]).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                if ema is not None: ema.update(model)

                running += loss.item(); n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, n_batches)

        # ---- Val (pairwise proxy loss for logging) ----
        model.eval()
        val_running, val_nb = 0.0, 0
        pbar_v = tqdm(pair_dl_val, desc=f"Epoch {epoch}/{args.epochs} [val pairs]", leave=False)
        with torch.no_grad():
            for x1,x2,y_pair,m1,m2 in pbar_v:
                x1=x1.to(device); x2=x2.to(device)
                y_pair=y_pair.float().to(device)
                m1 = m1.to(device) if m1.numel()>0 else None
                m2 = m2.to(device) if m2.numel()>0 else None
                with amp_autocast_for(device):
                    logits, z1, z2, masks = model.pair_forward(x1,x2,m1,m2)
                    if args.loss=="bce":
                        vloss = criterion_bce(logits, y_pair)
                    else:
                        vloss = contrastive_loss_from_embeddings(z1, z2, y_pair, margin=args.contrastive_margin)
                val_running += vloss.item(); val_nb += 1
                pbar_v.set_postfix(loss=f"{vloss.item():.4f}")
        val_pair_loss = val_running / max(1, val_nb)

        # ---- Prototype eval on VAL every N ----
        eval_model = ema.ema if ema is not None else model
        if args.embed_every>0 and (epoch % args.embed_every == 0):
            val_proto_acc, val_proto_auc = prototype_eval(
                eval_model, device, ds_train, ds_val,
                batch=args.embed_batch, max_items=args.embed_max
            )
        else:
            val_proto_acc, val_proto_auc = float("nan"), float("nan")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_pair_loss:.6f},{val_proto_acc:.6f},{val_proto_auc:.6f}\n")
        print(f"[{epoch:02d}/{args.epochs}] train={train_loss:.4f} | val_pair={val_pair_loss:.4f} | "
              f"proto_acc={val_proto_acc:.4f} | proto_auc={val_proto_auc:.4f}")

        # ---- Save every N ----
        if epoch % args.save_every == 0:
            ckpt_path = run_dir / f"siamese_epoch{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": (ema.ema.state_dict() if ema is not None else None),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(args)
            }, ckpt_path)
            print("Saved:", ckpt_path)

        if not math.isnan(val_proto_acc) and (val_proto_acc > best_acc):
            best_acc = val_proto_acc
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_state": (ema.ema.state_dict() if ema is not None else None)},
                       run_dir / "best.pt")

        # ---- TEST (balanced) cadence ----
        if epoch % args.test_every == 0:
            test_bal_acc, test_bal_auc = prototype_eval(
                eval_model, device, ds_train, ds_test_bal,
                batch=args.embed_batch, max_items=args.embed_max
            )
            with open(metrics_path, "a") as f:
                f.write(f"test_bal@{epoch},,,{test_bal_acc:.6f},{test_bal_auc:.6f}\n")
            print(f"[TEST-BAL@{epoch}] proto_acc={test_bal_acc:.4f} | proto_auc={test_bal_auc:.4f}")

            if test_bal_acc > best_test_acc:
                best_test_acc = test_bal_acc
                torch.save({"epoch": epoch,
                            "model_state": model.state_dict(),
                            "ema_state": (ema.ema.state_dict() if ema is not None else None)},
                           run_dir / "best_test.pt")

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

            if args.lr_decay_on_drop and (last_test_acc is not None):
                if test_bal_acc < (last_test_acc - args.drop_tolerance):
                    new_lr = adjust_lr_on_drop(optimizer, factor=args.lr_decay_factor, min_lr=args.lr_min)
                    print(f"[LR-DECAY] test acc {last_test_acc:.4f} → {test_bal_acc:.4f}. lr={new_lr:.2e}")
            last_test_acc = test_bal_acc

        # ---- (Optional) rebuild HNM pool for pair training ----
        if args.hnm and (args.loss in ["bce","contrastive"]) and (epoch >= args.hnm_start) and (epoch % args.hnm_every == 0):
            print("[HNM] Rebuilding pool ...")
            hnm_model = ema.ema if ema is not None else model
            pool = build_hard_negative_pool(
                hnm_model, device, ds_train,
                topk=args.hnm_topk, max_pos=args.hnm_max_pos, max_neg=args.hnm_max_neg,
                batch=args.hnm_embed_batch, seed=args.seed
            )
            pair_ds_train.set_hard_negatives(pool, frac=args.hnm_frac)
            print(f"[HNM] Pool for {len(pool)} positives; frac={args.hnm_frac:.2f}")

    # Final test (if not early stopped)
    if not stopped_early:
        eval_model = ema.ema if ema is not None else model
        test_bal_acc, test_bal_auc = prototype_eval(
            eval_model, device, ds_train, ds_test_bal,
            batch=args.embed_batch, max_items=args.embed_max
        )
        with open(metrics_path, "a") as f:
            f.write(f"test_bal@final,,,{test_bal_acc:.6f},{test_bal_auc:.6f}\n")
        print(f"[TEST-BAL@final] proto_acc={test_bal_acc:.4f} | proto_auc={test_bal_auc:.4f}")

    torch.save({"epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": (ema.ema.state_dict() if ema is not None else None)},
               run_dir / "last.pt")
    with open(run_dir / "config.json", "w") as f: json.dump(vars(args), f, indent=2)
    print("Run dir:", run_dir)
    print("Metrics:", metrics_path)

# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser("ISIC-2020 Siamese — triplet/contrastive, learned mask, metadata (fixed meta alignment)")
    p.add_argument("--data-root", type=str, default="", help="If empty, mirror is pulled via kagglehub.")
    p.add_argument("--out-dir", type=str, default="./runs")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--pn-pp-nn", type=float, nargs=3, default=(0.75,0.125,0.125), help="For BCE/contrastive.")
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--embed-every", type=int, default=1)
    p.add_argument("--embed-batch", type=int, default=512)
    p.add_argument("--embed-max", type=int, default=20000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--val-size",  type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu","mps"])

    # Eval cadence / early stop / LR decay on drops
    p.add_argument("--test-every", type=int, default=5)
    p.add_argument("--early-stop-acc", type=float, default=0.80)
    p.add_argument("--lr-decay-on-drop", action="store_true")
    p.add_argument("--drop-tolerance", type=float, default=0.003)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--lr-min", type=float, default=1e-6)

    # Loss & margins
    p.add_argument("--loss", choices=["bce","contrastive","triplet"], default="triplet")
    p.add_argument("--contrastive-margin", type=float, default=0.5)
    p.add_argument("--triplet-margin", type=float, default=0.2)

    # PN/PP/NN mix schedule (for BCE/contrastive)
    p.add_argument("--use-mix-schedule", action="store_true")
    p.add_argument("--mix-ep1", type=int, default=5)
    p.add_argument("--mix-ep2", type=int, default=20)

    # EMA
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.995)

    # HNM (only used for BCE/contrastive)
    p.add_argument("--hnm", action="store_true")
    p.add_argument("--hnm-start", type=int, default=8)
    p.add_argument("--hnm-every", type=int, default=1)
    p.add_argument("--hnm-frac", type=float, default=0.7)
    p.add_argument("--hnm-topk", type=int, default=10)
    p.add_argument("--hnm-max-pos", type=int, default=2000)
    p.add_argument("--hnm-max-neg", type=int, default=4000)
    p.add_argument("--hnm-embed-batch", type=int, default=512)

    # Learned mask
    p.add_argument("--mask-mode", choices=["off","learn"], default="off")
    p.add_argument("--mask-l1", type=float, default=1e-4, help="Sparsity reg on learned mask (if enabled).")

    # Metadata
    p.add_argument("--use-metadata", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
