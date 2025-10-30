
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import SiameseResNet18
from dataset import build_datasets


# ---- device helpers (minimal) ----
def choose_device(pref="auto"):
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ---- prototype scoring ----
@torch.no_grad()
def _compute_embeddings(model, device, ds, batch=512, max_items=None):
    from torch.utils.data import DataLoader, Subset
    if (max_items is not None) and (len(ds) > max_items):
        import numpy as _np
        idx = _np.random.choice(len(ds), size=max_items, replace=False).tolist()
        ds = Subset(ds, idx)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    all_z, all_y = [], []
    model.eval()
    for xb, yb in dl:
        xb = xb.to(device)
        z = model.forward_once(xb)
        all_z.append(z.cpu()); all_y.append(yb)
    return torch.cat(all_z, 0), torch.cat(all_y, 0)


@torch.no_grad()
def prototype_predict(eval_model, device, ds_train, ds_eval, batch=512, max_items=None):
    import torch.nn.functional as F
    z_tr, y_tr = _compute_embeddings(eval_model, device, ds_train, batch=batch, max_items=max_items)
    c0 = F.normalize(z_tr[y_tr==0].mean(0, keepdim=True), dim=1) if (y_tr==0).any() else torch.zeros_like(z_tr[:1])
    c1 = F.normalize(z_tr[y_tr==1].mean(0, keepdim=True), dim=1) if (y_tr==1).any() else torch.zeros_like(z_tr[:1])

    z_ev, y_ev = _compute_embeddings(eval_model, device, ds_eval, batch=batch, max_items=max_items)
    s1 = (z_ev * c1).sum(1)
    s0 = (z_ev * c0).sum(1)
    scores = (s1 - s0).cpu().numpy()  # higher => class 1
    y_true = y_ev.numpy().astype(int)
    y_pred = (scores > 0).astype(int)
    return y_true, y_pred, scores, z_ev.numpy()


def parse_args():
    p = argparse.ArgumentParser("Predict/evaluate Siamese (prototypes) with ROC & visualisations")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (*.pt).")
    p.add_argument("--data-root", type=str, default="")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--embed-batch", type=int, default=512)
    p.add_argument("--embed-max", type=int, default=20000)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu","mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class _ArgsShim:
    def __init__(self, ns):
        self.data_root = ns.data_root
        self.img_size = ns.img_size
        self.val_size = ns.val_size
        self.test_size = ns.test_size
        self.seed = ns.seed


def main():
    args = parse_args()
    device = choose_device(args.device)
    print("Device:", device)

    # datasets
    ds_train, _, ds_test_bal = build_datasets(_ArgsShim(args))

    # load model (& EMA if present)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SiameseResNet18().to(device)
    model.load_state_dict(ckpt["model_state"])
    if ckpt.get("ema_state") is not None:
        try:
            model.load_state_dict(ckpt["ema_state"], strict=False)
            print("[info] Using EMA weights from checkpoint.")
        except Exception:
            print("[warn] EMA present but could not be loaded; using model_state.")

    # predict (prototypes)
    y_true, y_pred, scores, z_test = prototype_predict(
        model, device, ds_train, ds_test_bal, batch=args.embed_batch, max_items=args.embed_max
    )

    # metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    try:
        auc_val = roc_auc_score(y_true, scores)
    except Exception:
        auc_val = float("nan")
    print(f"[BAL TEST] proto_acc={acc:.4f} | proto_auc={auc_val:.4f} | n={len(y_true)}")

    # output dir beside checkpoint
    ckpt_dir = Path(args.ckpt).resolve().parent
    viz_dir = ckpt_dir / "predict_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (balanced test)")
    plt.legend(); plt.tight_layout()
    roc_path = viz_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150); plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (balanced test)")
    cm_path = viz_dir / "confusion_matrix.png"
    plt.tight_layout(); plt.savefig(cm_path, dpi=150); plt.close()

    # PCA of embeddings (2D)
    if z_test.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=args.seed)
        z2 = pca.fit_transform(z_test)
        plt.figure()
        plt.scatter(z2[y_true==0,0], z2[y_true==0,1], s=8, label="class 0")
        plt.scatter(z2[y_true==1,0], z2[y_true==1,1], s=8, label="class 1")
        plt.legend(); plt.title("Test embeddings (PCA-2D)")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
        pca_path = viz_dir / "embeddings_pca.png"
        plt.savefig(pca_path, dpi=150); plt.close()
    else:
        pca_path = None

    print("Saved visualisations:")
    print(" -", roc_path)
    print(" -", cm_path)
    if pca_path: print(" -", pca_path)


if __name__ == "__main__":
    main()
