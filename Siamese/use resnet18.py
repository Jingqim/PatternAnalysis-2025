#%% DATA LOADING & PREPROCESSING (helpers only; NO side effects at import)
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchvision.io import read_image
from torchvision.models import resnet18  # NOT pretrained (weights=None)

from sklearn.model_selection import GroupShuffleSplit  # preprocessing only
from sklearn.metrics import roc_auc_score  # threshold-free metric

IMG_SIZE = 224  # shared image size
# ImageNet normalization (helps optimization even from scratch)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

class ISIC2020Dataset(Dataset):
    """
    - Reads RGB via torchvision.io.read_image -> [C,H,W] uint8
    - Converts to float [0,1], resizes to IMG_SIZE, normalizes, returns (image, label_float)
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
        # Normalize (channel-wise)
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
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
    pos_df = split_df[split_df["target"] == 1]
    neg_df = split_df[split_df["target"] == 0]
    k_pos = len(pos_df)
    if k_pos == 0:
        # fallback if no positives (unlikely for ISIC 2020)
        return neg_df.sample(n=min(1000, len(neg_df)), random_state=seed).reset_index(drop=True)
    k_neg = min(k_pos, len(neg_df))
    neg_sample = neg_df.sample(n=k_neg, random_state=seed)
    out = pd.concat([pos_df, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


#%% TRAINING — ResNet18 (from scratch), cosine+BCEWithLogits, warm-up, hard-mining, GPU checks
import multiprocessing as mp
from tqdm.auto import tqdm

# ---------------------------
# Encoder & Siamese (cosine)
# ---------------------------
EMB_DIM = 512  # resnet18 pooled feature dim

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)  # train from scratch
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        x = self.stem(x)            # [B,512,1,1]
        x = x.view(x.size(0), -1)   # [B,512]
        x = nn.functional.normalize(x, dim=1)
        return x

class SiameseCosine(nn.Module):
    """
    Cosine similarity + learnable temperature and bias, used with BCEWithLogitsLoss.
    logit = scale * cos + bias
    """
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.scale = nn.Parameter(torch.tensor(10.0))   # temperature
        self.bias  = nn.Parameter(torch.tensor(0.0))    # NEW: learnable bias
    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        cos = torch.sum(z1 * z2, dim=1)                 # [-1,1]
        logits = self.scale * cos + self.bias           # BCEWithLogitsLoss expects logits
        return logits

# ---------------------------------
# Balanced batch sampler (images)
# ---------------------------------
class BalancedBatchSampler(BatchSampler):
    """
    Always includes >= min_pos positives per batch (rest negatives).
    Yields lists of indices into the underlying dataset (train image set).
    """
    def __init__(self, labels: np.ndarray, batch_size: int, min_pos: int, seed: int = 0):
        self.labels = labels.astype(int)
        self.batch_size = batch_size
        self.min_pos = min_pos
        self.rng = np.random.default_rng(seed)
        self.pos_idx = np.where(self.labels == 1)[0].tolist()
        self.neg_idx = np.where(self.labels == 0)[0].tolist()
        assert batch_size > 1, "batch_size must be > 1"
        assert len(self.pos_idx) >= 1, "No positive samples found."

    def __iter__(self):
        pos = np.array(self.pos_idx)
        neg = np.array(self.neg_idx)
        self.rng.shuffle(pos); self.rng.shuffle(neg)
        pi = 0; ni = 0
        total = len(self.labels)
        n_batches = max(1, total // self.batch_size)
        for _ in range(n_batches):
            need_pos = min(self.min_pos, len(pos)) if len(pos) > 0 else 0
            if pi + need_pos > len(pos):
                self.rng.shuffle(pos); pi = 0
            pos_take = pos[pi:pi+need_pos]; pi += need_pos

            need_neg = self.batch_size - len(pos_take)
            if ni + need_neg > len(neg):
                self.rng.shuffle(neg); ni = 0
            neg_take = neg[ni:ni+need_neg]; ni += need_neg

            batch = np.concatenate([pos_take, neg_take])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return max(1, len(self.labels) // self.batch_size)

# ------------------------------------------------------------
# Pair sampling: hard-mined (original) and RANDOM warm-up
# ------------------------------------------------------------
def mine_pairs_from_batch(z: torch.Tensor, y: torch.Tensor,
                          ratio=(0.6, 0.2, 0.2),  # PN, PP, NN
                          max_pairs_per_batch: Optional[int] = None):
    """
    Hard mining using cosine (same as before).
    Returns indices (i_idx, j_idx) and pair labels y_pairs (1=same, 0=diff).
    """
    device = z.device
    B = z.size(0)
    with torch.no_grad():
        cos = z @ z.t()                       # [B,B], cosine
        _ = ~torch.eye(B, dtype=torch.bool, device=device)

    pos_idx = torch.where(y == 1)[0]
    neg_idx = torch.where(y == 0)[0]
    n_pos = len(pos_idx); n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        # fallback: random pairs
        return sample_pairs_from_batch_random(y, ratio=ratio, max_pairs_per_batch=max_pairs_per_batch)

    # available counts
    avail_pn = n_pos * n_neg
    avail_pp = n_pos * (n_pos - 1) // 2
    avail_nn = n_neg * (n_neg - 1) // 2

    # desired counts by ratio
    if max_pairs_per_batch is None:
        max_pairs_per_batch = min(4096, avail_pn + avail_pp + avail_nn)
    r = np.array(ratio, dtype=np.float64); r = r / r.sum()
    want_pn = int(r[0] * max_pairs_per_batch)
    want_pp = int(r[1] * max_pairs_per_batch)
    want_nn = int(r[2] * max_pairs_per_batch)

    # clamp + rebalance leftovers
    want_pn = min(want_pn, avail_pn)
    want_pp = min(want_pp, avail_pp)
    want_nn = min(want_nn, avail_nn)
    alloc = [want_pn, want_pp, want_nn]
    avail = [avail_pn, avail_pp, avail_nn]
    target_total = min(max_pairs_per_batch, sum(avail))
    while sum(alloc) < target_total:
        deficits = [(avail[i] - alloc[i], i) for i in range(3)]
        deficits.sort(reverse=True)
        for left, i in deficits:
            if left > 0:
                alloc[i] += 1
                break
        else:
            break
    want_pn, want_pp, want_nn = alloc

    # PN: pick highest cosine (hard negatives)
    pn_i, pn_j = [], []
    for i in pos_idx:
        c = cos[i, neg_idx]
        vals, idxs = torch.sort(c, descending=True)
        take = min(want_pn - sum(x.numel() for x in pn_i), len(idxs))
        if take > 0:
            pn_i.append(i.repeat(take))
            pn_j.append(neg_idx[idxs[:take]])
        if sum(x.numel() for x in pn_i) >= want_pn:
            break
    if pn_i:
        pn_i = torch.cat(pn_i); pn_j = torch.cat(pn_j)
    else:
        pn_i = torch.tensor([], dtype=torch.long, device=device)
        pn_j = torch.tensor([], dtype=torch.long, device=device)

    # PP: lowest cosine (hard positives)
    if n_pos >= 2 and want_pp > 0:
        c_pp = cos[pos_idx][:, pos_idx]
        triu = torch.triu(torch.ones_like(c_pp, dtype=torch.bool), diagonal=1)
        vals = c_pp[triu]
        vals_sorted, order = torch.sort(vals, descending=False)
        take = min(want_pp, vals_sorted.numel())
        coords = triu.nonzero(as_tuple=False)[order[:take]]
        pp_i = pos_idx[coords[:, 0]]
        pp_j = pos_idx[coords[:, 1]]
    else:
        pp_i = torch.tensor([], dtype=torch.long, device=device)
        pp_j = torch.tensor([], dtype=torch.long, device=device)

    # NN: lowest cosine (hard same-neg)
    if n_neg >= 2 and want_nn > 0:
        c_nn = cos[neg_idx][:, neg_idx]
        triu = torch.triu(torch.ones_like(c_nn, dtype=torch.bool), diagonal=1)
        vals = c_nn[triu]
        vals_sorted, order = torch.sort(vals, descending=False)
        take = min(want_nn, vals_sorted.numel())
        coords = triu.nonzero(as_tuple=False)[order[:take]]
        nn_i = neg_idx[coords[:, 0]]
        nn_j = neg_idx[coords[:, 1]]
    else:
        nn_i = torch.tensor([], dtype=torch.long, device=device)
        nn_j = torch.tensor([], dtype=torch.long, device=device)

    i_idx = torch.cat([pn_i, pp_i, nn_i]) if pn_i.numel()+pp_i.numel()+nn_i.numel() > 0 \
            else torch.tensor([], dtype=torch.long, device=device)
    j_idx = torch.cat([pn_j, pp_j, nn_j]) if pn_j.numel()+pp_j.numel()+nn_j.numel() > 0 \
            else torch.tensor([], dtype=torch.long, device=device)
    y_pairs = torch.cat([
        torch.zeros(pn_i.numel(), dtype=torch.float32, device=device),
        torch.ones(pp_i.numel()+nn_i.numel(), dtype=torch.float32, device=device)
    ]) if i_idx.numel() > 0 else torch.tensor([], dtype=torch.float32, device=device)

    return i_idx, j_idx, y_pairs

def sample_pairs_from_batch_random(y: torch.Tensor,
                                   ratio=(0.6, 0.2, 0.2),
                                   max_pairs_per_batch: Optional[int] = None):
    """
    RANDOM pair sampling (no mining) used for warm-up.
    Builds PN, PP, NN by random choice according to ratio and availability.
    """
    device = y.device
    pos_idx = torch.where(y == 1)[0]
    neg_idx = torch.where(y == 0)[0]
    n_pos = int(pos_idx.numel()); n_neg = int(neg_idx.numel())

    avail_pn = n_pos * n_neg
    avail_pp = n_pos * (n_pos - 1) // 2
    avail_nn = n_neg * (n_neg - 1) // 2
    if max_pairs_per_batch is None:
        max_pairs_per_batch = min(4096, avail_pn + avail_pp + avail_nn)

    r = np.array(ratio, dtype=np.float64); r = r / r.sum()
    want_pn = min(int(r[0] * max_pairs_per_batch), avail_pn)
    want_pp = min(int(r[1] * max_pairs_per_batch), avail_pp)
    want_nn = min(int(r[2] * max_pairs_per_batch), avail_nn)

    # Rebalance leftover capacity
    alloc = [want_pn, want_pp, want_nn]; avail = [avail_pn, avail_pp, avail_nn]
    target_total = min(max_pairs_per_batch, sum(avail))
    while sum(alloc) < target_total:
        deficits = [(avail[i] - alloc[i], i) for i in range(3)]
        deficits.sort(reverse=True)
        for left, i in deficits:
            if left > 0:
                alloc[i] += 1
                break
        else:
            break
    want_pn, want_pp, want_nn = alloc

    i_list, j_list, y_list = [], [], []

    # PN: random all pos-neg combinations
    if want_pn > 0 and n_pos > 0 and n_neg > 0:
        # Sample with replacement if needed (shouldn't happen post clamp)
        p_idx = pos_idx[torch.randint(0, n_pos, (want_pn,), device=device)]
        n_idx = neg_idx[torch.randint(0, n_neg, (want_pn,), device=device)]
        i_list.append(p_idx); j_list.append(n_idx)
        y_list.append(torch.zeros(want_pn, dtype=torch.float32, device=device))

    # PP: random unique pairs approx via random pairing (small bias OK for warm-up)
    if want_pp > 0 and n_pos >= 2:
        a = pos_idx[torch.randint(0, n_pos, (want_pp,), device=device)]
        b = pos_idx[torch.randint(0, n_pos, (want_pp,), device=device)]
        mask = a != b
        a, b = a[mask], b[mask]
        take = min(int(a.numel()), want_pp)
        i_list.append(a[:take]); j_list.append(b[:take])
        y_list.append(torch.ones(take, dtype=torch.float32, device=device))

    # NN: random pairs among negatives
    if want_nn > 0 and n_neg >= 2:
        a = neg_idx[torch.randint(0, n_neg, (want_nn,), device=device)]
        b = neg_idx[torch.randint(0, n_neg, (want_nn,), device=device)]
        mask = a != b
        a, b = a[mask], b[mask]
        take = min(int(a.numel()), want_nn)
        i_list.append(a[:take]); j_list.append(b[:take])
        y_list.append(torch.ones(take, dtype=torch.float32, device=device))

    if len(i_list) == 0:
        return (torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.float32, device=device))

    i_idx = torch.cat(i_list); j_idx = torch.cat(j_list); y_pairs = torch.cat(y_list)
    return i_idx, j_idx, y_pairs


# ---------------------------
# Eval helpers (balanced) + metrics & threshold search
# ---------------------------
def _compute_metrics_from_logits_np(logits: np.ndarray,
                                    y_true: np.ndarray,
                                    fpr_target: float = 0.10,
                                    grid_size: int = 199):
    """
    Threshold-free AUC + threshold search over quantiles of logits.
    Returns dict: auc, acc@0, best acc & t, best F1 & t, best TPR under FPR<=target & t.
    """
    assert logits.ndim == 1 and y_true.ndim == 1
    n = y_true.size
    try:
        auc = float(roc_auc_score(y_true, logits)) if (np.unique(y_true).size > 1) else float("nan")
    except Exception:
        auc = float("nan")
    acc_at_0 = float(((logits >= 0).astype(np.float32) == y_true).mean()) if n > 0 else 0.0

    if n == 0:
        return dict(auc=auc, acc0=acc_at_0,
                    acc_best=acc_at_0, t_acc=0.0,
                    f1_best=0.0,  t_f1=0.0,
                    tpr_best=0.0, t_tpr=0.0)

    qs = np.linspace(0.0, 1.0, grid_size + 2)[1:-1]
    thrs = np.unique(np.quantile(logits, qs))

    best_acc = -1.0; t_acc = 0.0
    best_f1  = -1.0; t_f1  = 0.0
    best_tpr = -1.0; t_tpr = 0.0

    pos_mask = (y_true == 1)
    neg_mask = ~pos_mask
    P = float(pos_mask.sum())
    N = float(neg_mask.sum())

    for t in thrs:
        pred = (logits >= t)
        tp = float((pred & pos_mask).sum())
        tn = float((~pred & neg_mask).sum())
        fp = float((pred & neg_mask).sum())
        fn = float((~pred & pos_mask).sum())
        acc = (tp + tn) / n
        if acc > best_acc:
            best_acc = acc; t_acc = float(t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1; t_f1 = float(t)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = rec
        if fpr <= fpr_target and tpr >= max(best_tpr, 0.0):
            best_tpr = tpr; t_tpr = float(t)

    if best_tpr < 0:
        best_tpr, t_tpr = 0.0, float(np.median(logits))

    return dict(
        auc=float(auc),
        acc0=float(acc_at_0),
        acc_best=float(best_acc), t_acc=float(t_acc),
        f1_best=float(best_f1),  t_f1=float(t_f1),
        tpr_best=float(best_tpr), t_tpr=float(t_tpr),
    )

def eval_on_pairs(model: nn.Module,
                  loader: DataLoader,
                  device: torch.device,
                  desc: str = "Eval",
                  leave: bool = False,
                  fpr_target: float = 0.10,
                  grid_size: int = 199) -> Tuple[float, dict]:
    """Evaluate on a fixed pair dataset. Returns (mean_loss, metrics_dict)."""
    model.eval()
    bce_logits = nn.BCEWithLogitsLoss()
    totL = n_batches = 0
    all_logits = []; all_y = []

    with torch.no_grad():
        for x1, x2, y in tqdm(loader, desc=desc, leave=leave):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = y.to(device)
            logits = model(x1, x2)
            L = bce_logits(logits, y)
            totL += float(L.item()); n_batches += 1
            all_logits.append(logits.detach().float().cpu())
            all_y.append(y.detach().float().cpu())

    mean_loss = totL / max(1, n_batches)
    if all_logits:
        logits_np = torch.cat(all_logits).numpy()
        y_np = torch.cat(all_y).numpy()
        metrics = _compute_metrics_from_logits_np(logits_np, y_np,
                                                  fpr_target=fpr_target,
                                                  grid_size=grid_size)
    else:
        metrics = dict(auc=float("nan"), acc0=0.0,
                       acc_best=0.0, t_acc=0.0,
                       f1_best=0.0, t_f1=0.0,
                       tpr_best=0.0, t_tpr=0.0)
    return mean_loss, metrics


# ---------------------------
# Main training entrypoint
# ---------------------------
def main():
    # --- Config ---
    CSV_PATH = Path("./train-metadata.csv")
    IMG_DIR  = Path("./train-image/image")

    NUM_WORKERS = 4
    BATCH_SIZE  = 64
    MIN_POS_PER_BATCH = 8
    EPOCHS = 50
    MAX_EVAL_PAIRS = 10000

    # Validation metrics configs
    FPR_TARGET = 0.10     # report TPR at ≤10% FPR
    GRID_SIZE  = 199      # ~200 thresholds over logits

    # Warm-up schedule
    WARMUP_EPOCHS = 5
    RATIO_WARMUP  = (0.4, 0.4, 0.2)  # PN:PP:NN during warm-up (easier)
    RATIO_MINED   = (0.6, 0.2, 0.2)  # after warm-up (original)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = torch.cuda.is_available()
    PERSISTENT = True
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    print(f"[info] Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[cuda] {torch.cuda.get_device_name(0)}")

    # --- Prepare splits ---
    train_df, val_df, test_df = prepare_splits(CSV_PATH, IMG_DIR)

    # --- Datasets ---
    train_ds = ISIC2020Dataset(train_df)
    train_labels = train_df["target"].to_numpy().astype(int)

    val_bal_df  = make_balanced_subset(val_df,  seed=43)
    test_bal_df = make_balanced_subset(test_df, seed=44)
    print("Balanced val images:", len(val_bal_df),  "| pos:", int((val_bal_df['target']==1).sum()),
          "| neg:", int((val_bal_df['target']==0).sum()))
    print("Balanced test images:", len(test_bal_df), "| pos:", int((test_bal_df['target']==1).sum()),
          "| neg:", int((test_bal_df['target']==0).sum()))

    val_ds  = ISIC2020Dataset(val_bal_df)
    test_ds = ISIC2020Dataset(test_bal_df)

    # --- Fixed PAIR sets for eval ---
    val_pairs_idx,  val_pairs_y  = make_balanced_pair_indices(val_bal_df["target"].to_numpy(),  max_pairs=MAX_EVAL_PAIRS, seed=1)
    test_pairs_idx, test_pairs_y = make_balanced_pair_indices(test_bal_df["target"].to_numpy(), max_pairs=MAX_EVAL_PAIRS, seed=2)
    val_pair_ds  = FixedPairsDataset(val_ds,  val_pairs_idx,  val_pairs_y)
    test_pair_ds = FixedPairsDataset(test_ds, test_pairs_idx, test_pairs_y)

    # --- DataLoaders ---
    batch_sampler = BalancedBatchSampler(labels=train_labels, batch_size=BATCH_SIZE, min_pos=MIN_POS_PER_BATCH, seed=123)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)
    val_loader   = DataLoader(val_pair_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)
    test_loader  = DataLoader(test_pair_ds, batch_size=64, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT)

    # --- Model / loss / optim / LR DECAY ---
    model = SiameseCosine().to(DEVICE)
    bce_logits = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 3e-4},   # encoder
        {"params": [model.scale, model.bias], "lr": 1e-3},    # temperature + bias
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # --- Train ---
    first_cuda_check_done = False
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        runL = runA = n = 0

        use_warmup = (epoch <= WARMUP_EPOCHS)
        ratio_now = RATIO_WARMUP if use_warmup else RATIO_MINED

        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE)

            if (not first_cuda_check_done) and (DEVICE.type == "cuda"):
                assert imgs.is_cuda, "Images are not on CUDA!"
                print("[check] CUDA is in use ✅")
                first_cuda_check_done = True

            # Encode once per image-batch
            z = model.encoder(imgs)  # [B,512], normalized

            if use_warmup:
                i_idx, j_idx, y_pairs = sample_pairs_from_batch_random(
                    labels, ratio=ratio_now, max_pairs_per_batch=BATCH_SIZE*4
                )
            else:
                i_idx, j_idx, y_pairs = mine_pairs_from_batch(
                    z, labels, ratio=ratio_now, max_pairs_per_batch=BATCH_SIZE*4
                )
            if i_idx.numel() == 0:
                continue

            # Logits from cosine via model.scale + bias (keep graph for encoder/scale/bias)
            cos_vals = torch.sum(z[i_idx] * z[j_idx], dim=1)    # [P]
            logits = model.scale * cos_vals + model.bias        # [P]
            loss = bce_logits(logits, y_pairs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = ((logits >= 0).float() == y_pairs).float().mean().item()  # same as p>=0.5

            n += 1
            runL += loss.item()
            runA += acc
            pbar.set_postfix(loss=f"{runL/n:.4f}", acc=f"{runA/n:.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # Validation (ROC-AUC + threshold search)
        vL, vM = eval_on_pairs(model, val_loader, DEVICE,
                               desc=f"Validate (epoch {epoch})", leave=False,
                               fpr_target=FPR_TARGET, grid_size=GRID_SIZE)

        print(
            f"[epoch {epoch}] "
            f"train_loss={runL/max(1,n):.4f} "
            f"train_pair_acc(hard)={runA/max(1,n):.3f} | "
            f"val_loss={vL:.4f} "
            f"val_auc={vM['auc']:.3f} "
            f"val_acc@0={vM['acc0']:.3f} | "
            f"val_acc@t*={vM['acc_best']:.3f} (t*={vM['t_acc']:.4f}) "
            f"val_f1@t*={vM['f1_best']:.3f} (t*={vM['t_f1']:.4f}) "
            f"val_tpr@fpr<={FPR_TARGET:.2f}={vM['tpr_best']:.3f} (t={vM['t_tpr']:.4f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    # --- Final Test ---
    tL, tM = eval_on_pairs(model, test_loader, DEVICE, desc="Test", leave=True,
                           fpr_target=FPR_TARGET, grid_size=GRID_SIZE)
    print(
        f"[test] loss={tL:.4f} "
        f"auc={tM['auc']:.3f} "
        f"acc@0={tM['acc0']:.3f} "
        f"acc@t*={tM['acc_best']:.3f} (t*={tM['t_acc']:.4f}) "
        f"f1@t*={tM['f1_best']:.3f} (t*={tM['t_f1']:.4f}) "
        f"tpr@fpr<={FPR_TARGET:.2f}={tM['tpr_best']:.3f} (t={tM['t_tpr']:.4f})"
    )


# ---------------------------
# Fixed-pair utilities (unchanged)
# ---------------------------
def make_balanced_pair_indices(labels: np.ndarray, max_pairs: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    same_pairs = []
    if len(pos) >= 2:
        same_pairs.append(np.array([(int(pos[a]), int(pos[b])) for a in range(len(pos)) for b in range(a+1, len(pos))], dtype=np.int32))
    if len(neg) >= 2:
        same_pairs.append(np.array([(int(neg[a]), int(neg[b])) for a in range(len(neg)) for b in range(a+1, len(neg))], dtype=np.int32))
    same_pairs = np.concatenate(same_pairs, axis=0) if same_pairs else np.empty((0,2), dtype=np.int32)

    diff_pairs = np.array([(int(i), int(j)) for i in pos for j in neg], dtype=np.int32)

    half = max_pairs // 2
    take_s = min(len(same_pairs), half)
    take_d = min(len(diff_pairs), half)
    sel_same = same_pairs[rng.choice(len(same_pairs), size=take_s, replace=False)] if take_s > 0 else np.empty((0,2), dtype=np.int32)
    sel_diff = diff_pairs[rng.choice(len(diff_pairs), size=take_d, replace=False)] if take_d > 0 else np.empty((0,2), dtype=np.int32)

    pairs = np.vstack([sel_same, sel_diff]) if (take_s + take_d) > 0 else np.empty((0,2), dtype=np.int32)
    rng.shuffle(pairs)
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Windows-safe for DataLoader workers
    main()
