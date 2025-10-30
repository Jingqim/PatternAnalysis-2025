# train.py
import os, math, json, random, argparse, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from dataset import build_datasets, PairDataset
from modules import SiameseResNet18, ModelEMA


# ---------- AMP helpers ----------
def has_torch_amp():
    return hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def amp_autocast_for(device):
    if device.type != "cuda":
        class Dummy:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        return Dummy()
    if has_torch_amp():
        return torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        return torch.cuda.amp.autocast(enabled=True)

if has_torch_amp() and hasattr(torch.amp, "GradScaler"):
    GradScaler = torch.amp.GradScaler
else:
    GradScaler = torch.cuda.amp.GradScaler


# ---------- device & logging ----------
def choose_device(device_pref: str = "auto"):
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
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


# ---------- proto eval & HNM (minimal for training) ----------
@torch.no_grad()
def _maybe_subset(ds, max_items):
    if (max_items is None) or (len(ds) <= max_items): return ds
    idx = np.random.choice(len(ds), size=max_items, replace=False)
    from torch.utils.data import Subset
    return Subset(ds, idx.tolist())

@torch.no_grad()
def compute_embeddings(model, device, ds, batch=512, max_items=None):
    from torch.utils.data import DataLoader
    ds_sub = _maybe_subset(ds, max_items)
    dl = DataLoader(ds_sub, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    all_z, all_y = [], []
    model.eval()
    for xb, yb in tqdm(dl, desc="Embed", leave=False):
        xb = xb.to(device, non_blocking=False)
        with amp_autocast_for(device):
            z = model.forward_once(xb)
        all_z.append(z.cpu()); all_y.append(yb)
    import torch as _t
    return _t.cat(all_z, 0), _t.cat(all_y, 0)

@torch.no_grad()
def prototype_eval(eval_model, device, train_ds, val_or_test_ds, batch=512, max_items=None):
    import torch as _t
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score, accuracy_score

    z_tr, y_tr = compute_embeddings(eval_model, device, train_ds, batch=batch, max_items=max_items)
    c0 = F.normalize(z_tr[y_tr==0].mean(0, keepdim=True), dim=1) if (y_tr==0).any() else _t.zeros_like(z_tr[:1])
    c1 = F.normalize(z_tr[y_tr==1].mean(0, keepdim=True), dim=1) if (y_tr==1).any() else _t.zeros_like(z_tr[:1])

    z_va, y_va = compute_embeddings(eval_model, device, val_or_test_ds, batch=batch, max_items=max_items)
    s1 = (z_va * c1).sum(1)
    s0 = (z_va * c0).sum(1)
    scores = (s1 - s0).cpu().numpy()
    y_true = y_va.numpy()
    y_pred = (scores > 0).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    return acc, auc


@torch.no_grad()
def build_hard_negative_pool(eval_model, device, ds, topk=10, max_pos=2000, max_neg=4000, batch=512, seed=42):
    import torch as _t
    rng = np.random.default_rng(seed)
    labels = ds.df["target"].astype(int).values
    pos_idx = np.where(labels==1)[0]
    neg_idx = np.where(labels==0)[0]
    if len(pos_idx)==0 or len(neg_idx)==0:
        return {}
    pos_sel = rng.choice(pos_idx, size=min(len(pos_idx), max_pos), replace=False)
    neg_sel = rng.choice(neg_idx, size=min(len(neg_idx), max_neg), replace=False)

    from torch.utils.data import Subset, DataLoader
    zs_pos, zs_neg = [], []
    for idxs, bucket, tag in [(pos_sel, zs_pos, "HNM+"), (neg_sel, zs_neg, "HNM-")]:
        dl = DataLoader(Subset(ds, idxs.tolist()), batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
        for xb, _ in tqdm(dl, desc=f"{tag} embed", leave=False):
            xb = xb.to(device, non_blocking=False)
            with amp_autocast_for(device):
                z = eval_model.forward_once(xb)
            bucket.append(z.cpu())
    z_pos = torch.cat(zs_pos, 0)
    z_neg = torch.cat(zs_neg, 0)
    sim = z_pos @ z_neg.T
    k = min(topk, z_neg.shape[0])
    _, idxs = torch.topk(sim, k=k, dim=1, largest=True)
    pool = {}
    for row, pos_i in enumerate(pos_sel):
        hard_negs = neg_sel[idxs[row].cpu().numpy()].tolist()
        pool[int(pos_i)] = hard_negs
    return pool


# ---------- helper: plotting ----------
def _save_training_plots(hist, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Loss curves
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_pair_loss"], label="val_pair_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Losses")
    plt.tight_layout()
    f1 = out_dir / "loss_curves.png"
    plt.savefig(f1, dpi=150); plt.close()

    # Metrics
    plt.figure()
    plt.plot(hist["epoch"], hist["val_proto_acc"], label="val_proto_acc")
    if any(np.isfinite(hist["test_bal_acc"])):
        plt.plot(hist["epoch"], hist["test_bal_acc"], "o", label="test_bal_acc (points)")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Prototype Accuracy")
    plt.tight_layout()
    f2 = out_dir / "accuracy_curves.png"
    plt.savefig(f2, dpi=150); plt.close()

    plt.figure()
    plt.plot(hist["epoch"], hist["val_proto_auc"], label="val_proto_auc")
    if any(np.isfinite(hist["test_bal_auc"])):
        plt.plot(hist["epoch"], hist["test_bal_auc"], "o", label="test_bal_auc (points)")
    plt.xlabel("epoch"); plt.ylabel("AUC"); plt.legend(); plt.title("Prototype AUC")
    plt.tight_layout()
    f3 = out_dir / "auc_curves.png"
    plt.savefig(f3, dpi=150); plt.close()

    print("Saved plots:", f1, f2, f3)


# ---------- mix schedule ----------
def maybe_apply_mix_schedule(pair_ds: PairDataset, epoch: int, args):
    if not args.use_mix_schedule:
        pair_ds.set_mix(*tuple(args.pn_pp_nn))
        return
    if epoch <= args.mix_ep1:
        mix = (0.80, 0.10, 0.10)
    elif epoch <= args.mix_ep2:
        mix = (0.70, 0.15, 0.15)
    else:
        mix = (0.60, 0.20, 0.20)
    pair_ds.set_mix(*mix)


# ---------- training ----------
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
        f.write("epoch,train_loss,val_pair_loss,val_proto_acc,val_proto_auc,test_bal_acc,test_bal_auc,lr\n")

    # Data
    ds_train, ds_val, ds_test_bal = build_datasets(args)

    # Pair loaders
    pair_ds_train = PairDataset(ds_train, pn_pp_nn=tuple(args.pn_pp_nn), length=len(ds_train))
    pair_ds_val   = PairDataset(ds_val,   pn_pp_nn=(0.5, 0.25, 0.25), length=len(ds_val))
    maybe_apply_mix_schedule(pair_ds_train, 1, args)
    pair_dl_train = DataLoader(pair_ds_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=(device.type == "cuda"))
    pair_dl_val   = DataLoader(pair_ds_val,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=(device.type == "cuda"))
    print("Pair loaders:", len(pair_dl_train), len(pair_dl_val))

    # Model / Opt / Loss / EMA
    model = SiameseResNet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    best_acc = -1.0
    best_test_acc = -1.0
    last_test_acc = None
    stopped_early = False

    # history (for plots)
    hist = {"epoch": [], "train_loss": [], "val_pair_loss": [],
            "val_proto_acc": [], "val_proto_auc": [],
            "test_bal_acc": [], "test_bal_auc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        # ----- mix schedule -----
        maybe_apply_mix_schedule(pair_ds_train, epoch, args)

        # -------- Train --------
        model.train()
        running, n_batches = 0.0, 0
        pbar = tqdm(pair_dl_train, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for x1, x2, y in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = torch.as_tensor(y, dtype=torch.float32, device=device)

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
                y  = torch.as_tensor(y, dtype=torch.float32, device=device)
                with amp_autocast_for(device):
                    logits, _, _ = model(x1, x2)
                    vloss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                val_running += vloss.item(); val_nb += 1
                pbar_v.set_postfix(loss=f"{vloss.item():.4f}")
        val_pair_loss = val_running / max(1, val_nb)

        # -------- Prototype eval --------
        eval_model = ema.ema if ema is not None else model
        if args.embed_every > 0 and (epoch % args.embed_every == 0):
            val_proto_acc, val_proto_auc = prototype_eval(
                eval_model, device, ds_train, ds_val,
                batch=args.embed_batch, max_items=args.embed_max
            )
        else:
            val_proto_acc, val_proto_auc = float("nan"), float("nan")

        # -------- Periodic TEST (balanced) --------
        test_bal_acc = float("nan"); test_bal_auc = float("nan")
        if epoch % args.test_every == 0:
            test_bal_acc, test_bal_auc = prototype_eval(
                eval_model, device, ds_train, ds_test_bal,
                batch=args.embed_batch, max_items=args.embed_max
            )
            # Save best-test model
            if test_bal_acc > best_test_acc:
                best_test_acc = test_bal_acc
                torch.save({"epoch": epoch,
                            "model_state": model.state_dict(),
                            "ema_state": (ema.ema.state_dict() if ema is not None else None)},
                           run_dir / "best_test.pt")
            # Early stop
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

            # LR decay on drop (optional)
            if args.lr_decay_on_drop and (last_test_acc is not None):
                if test_bal_acc < (last_test_acc - args.drop_tolerance):
                    for pg in optimizer.param_groups:
                        pg["lr"] = max(pg["lr"] * args.lr_decay_factor, args.lr_min)
                    print(f"[LR-DECAY] test acc {last_test_acc:.4f} → {test_bal_acc:.4f}; new lr={optimizer.param_groups[0]['lr']:.2e}")
            last_test_acc = test_bal_acc

        # -------- Save checkpts / logs --------
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

        if not math.isnan(val_proto_acc) and (val_proto_acc > best_acc):
            best_acc = val_proto_acc
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_state": (ema.ema.state_dict() if ema is not None else None)},
                       run_dir / "best.pt")

        # log row
        cur_lr = optimizer.param_groups[0]["lr"]
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_pair_loss:.6f},{val_proto_acc:.6f},{val_proto_auc:.6f},{test_bal_acc:.6f},{test_bal_auc:.6f},{cur_lr:.6e}\n")

        # stash hist for plots
        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_pair_loss"].append(val_pair_loss)
        hist["val_proto_acc"].append(val_proto_acc)
        hist["val_proto_auc"].append(val_proto_auc)
        hist["test_bal_acc"].append(test_bal_acc)
        hist["test_bal_auc"].append(test_bal_auc)
        hist["lr"].append(cur_lr)

        # update plots each epoch
        _save_training_plots(hist, run_dir)

        if stopped_early:
            break

    # final eval + saves
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema.state_dict() if ema is not None else None)
    }, run_dir / "last.pt")
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Run dir:", run_dir)
    print("Metrics:", metrics_path)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Siamese ISIC2020 — scratch + schedule + EMA + HNM (+ plots)")
    p.add_argument("--data-root", type=str, default="", help="Dataset root (empty => kagglehub mirror).")
    p.add_argument("--out-dir", type=str, default="./runs", help="Dir to save logs/ckpts.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--pn-pp-nn", type=float, nargs=3, default=(0.75, 0.125, 0.125))
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--embed-every", type=int, default=1, help="Prototype eval cadence (0 disables).")
    p.add_argument("--embed-batch", type=int, default=512)
    p.add_argument("--embed-max", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4, help="Windows? set 0 if you see issues.")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu","mps"])

    p.add_argument("--test-every", type=int, default=1)
    p.add_argument("--early-stop-acc", type=float, default=0.80)
    p.add_argument("--lr-decay-on-drop", action="store_true")
    p.add_argument("--drop-tolerance", type=float, default=0.003)
    p.add_argument("--lr-decay-factor", type=float, default=0.25)
    p.add_argument("--lr-min", type=float, default=5e-6)

    p.add_argument("--use-mix-schedule", action="store_true")
    p.add_argument("--mix-ep1", type=int, default=5)
    p.add_argument("--mix-ep2", type=int, default=20)

    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.995)

    p.add_argument("--hnm", action="store_true")
    p.add_argument("--hnm-start", type=int, default=8)
    p.add_argument("--hnm-every", type=int, default=1)
    p.add_argument("--hnm-frac", type=float, default=0.7)
    p.add_argument("--hnm-topk", type=int, default=10)
    p.add_argument("--hnm-max-pos", type=int, default=2000)
    p.add_argument("--hnm-max-neg", type=int, default=4000)
    p.add_argument("--hnm-embed-batch", type=int, default=512)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
