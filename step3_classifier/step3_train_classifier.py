# step3_train_classifier.py
# Step 3: ResNet classifier with 2-channel input [recon_gray, input_gray]
# Positives (label=1): Step-1 synthetic anomalies  (step1_outputs/<site>/images)
# Negatives (label=0): dataset/<site>/train/defect-free
# Eval on real dataset/<site>/test (defect-free vs defect)
#
# Outputs:
#   <experiment_dir>/best_clf.pt
#   <experiment_dir>/test_metrics.json
#   <experiment_dir>/confusion_matrix.png

import argparse, json, math, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full determinism (may slow training a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- UNet (same as step2) ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2); dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNetSmall(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.inc   = DoubleConv(3, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.up1   = Up(base*4 + base*2, base*2)
        self.up2   = Up(base*2 + base, base)
        self.outc  = nn.Conv2d(base, 3, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x  = self.up1(x3, x2); x = self.up2(x, x1)
        return torch.sigmoid(self.outc(x))

# ---------------- IO helpers ----------------
IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def list_images(folder: Path):
    files = []
    for e in IMG_EXTS: files += list(folder.glob(f"*{e}"))
    return sorted(files)

def load_rgb(p: Path, size: Tuple[int,int]) -> torch.Tensor:
    im = Image.open(p).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    return TF.to_tensor(im)  # [0,1] CxHxW

def to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: Bx3xHxW in [0,1]; return Bx1xHxW
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    return 0.2989*r + 0.5870*g + 0.1140*b

# ---------------- Datasets ----------------
class TrainPairs(Dataset):
    """Training items come from:
       - positives: step1_outputs/<site>/images (label=1)
       - negatives: data_root/<site>/train/defect-free (label=0)
       We return raw RGB here; recon/2ch is built on-the-fly.
    """
    def __init__(self, data_root: Path, step1_root: Path, site: str, resize=(256,256)):
        self.resize = resize
        self.pos = list_images(Path(step1_root)/site/"images")            # label=1
        self.neg = list_images(Path(data_root)/site/"train"/"defect-free")# label=0
        if len(self.pos)==0 or len(self.neg)==0:
            raise SystemExit("No training images found for positives or negatives.")
        k = min(len(self.pos), len(self.neg))
        random.seed(42)
        self.items = [(p,1) for p in random.sample(self.pos, k)] + \
                    [(n,0) for n in random.sample(self.neg, k)]
        random.shuffle(self.items)

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, label = self.items[idx]
        x = load_rgb(p, self.resize)
        return x, label

class TestPairs(Dataset):
    """Evaluation on real test split."""
    def __init__(self, data_root: Path, site: str, resize=(256,256)):
        self.resize = resize
        root = Path(data_root)/site/"test"
        self.norm = [(p,0) for p in list_images(root/"defect-free")]
        self.defe = [(p,1) for p in list_images(root/"defect")]
        self.items = self.norm + self.defe
        if len(self.norm)==0 or len(self.defe)==0:
            raise SystemExit("No test images found.")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, label = self.items[idx]
        x = load_rgb(p, self.resize)
        return x, label, p.name

# ---------------- Build 2-channel input ----------------
@torch.no_grad()
def make_pair_batch(x_rgb: torch.Tensor, netR: nn.Module, device="cpu") -> torch.Tensor:
    """x_rgb: Bx3xHxW (0..1). Returns Bx2xHxW = [recon_gray, input_gray]."""
    y = netR(x_rgb)              # recon
    xin = to_gray(x_rgb)
    yre = to_gray(y)
    res = (xin - yre).abs() * 5.0   # amplify tiny differences; clamp below
    twoch = torch.cat([res.clamp(0,1), xin], dim=1)
    return torch.clamp(twoch, 0.0, 1.0)

# ---------------- ResNet18 with 2-channel stem ----------------
def build_resnet18_2ch(num_classes=2):
    m = resnet18(weights=None)  # train from scratch
    old = m.conv1
    m.conv1 = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
    nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------------- Train / Eval ----------------
def train_epoch(dl, netR, clf, opt, loss_fn, device):
    clf.train(); netR.eval()
    tot, n = 0.0, 0
    for x_rgb, y_lab in dl:
        x_rgb = x_rgb.to(device, non_blocking=True)
        y_lab = torch.as_tensor(y_lab, device=device, dtype=torch.long)

        twoch = make_pair_batch(x_rgb, netR, device=device)
        logits = clf(twoch)
        loss = loss_fn(logits, y_lab)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)  # guard
        opt.step()

        tot += loss.item() * x_rgb.size(0); n += x_rgb.size(0)
    return tot / max(1, n)

@torch.no_grad()
def eval_loss(dl, netR, clf, loss_fn, device):
    clf.eval(); netR.eval()
    tot, n = 0.0, 0
    for x_rgb, y_lab in dl:
        x_rgb = x_rgb.to(device)
        y_lab = y_lab.to(device)
        twoch = make_pair_batch(x_rgb, netR, device=device)
        logits = clf(twoch)
        tot += loss_fn(logits, y_lab).item() * x_rgb.size(0); n += x_rgb.size(0)
    return tot / max(1, n)

@torch.no_grad()
def evaluate(dl, netR, clf, device, threshold=0.5):
    clf.eval(); netR.eval()
    y_true, y_pred = [], []
    tp=fp=tn=fn=0
    for x_rgb, y_lab, _ in dl:
        x_rgb = x_rgb.to(device); y_lab = y_lab.to(device)
        twoch = make_pair_batch(x_rgb, netR, device=device)
        probs = torch.softmax(clf(twoch), dim=1)[:,1]  # P(defect)
        preds = (probs >= threshold).long()
        y_true.extend(y_lab.tolist()); y_pred.extend(preds.tolist())
        tp += ((preds==1)&(y_lab==1)).sum().item()
        fp += ((preds==1)&(y_lab==0)).sum().item()
        tn += ((preds==0)&(y_lab==0)).sum().item()
        fn += ((preds==0)&(y_lab==1)).sum().item()
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec = tp/max(1,(tp+fp))
    rec  = tp/max(1,(tp+fn))
    f1   = (2*prec*rec/max(1e-12, (prec+rec))) if (prec+rec)>0 else 0.0
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn,
                y_true=y_true, y_pred=y_pred)

def plot_confusion(cm: np.ndarray, labels: Tuple[str,str], out_path: Path):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    # grid + text
    ax.set_xticks(np.arange(-.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    # annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

# ---------------- Main ----------------
def main():
    set_seed(42)
    ap = argparse.ArgumentParser(description="Step 3: 2-ch ResNet on [recon_gray, input_gray]")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--step1_root", required=True)
    ap.add_argument("--site", choices=["SP3","SP5"], default="SP3")
    ap.add_argument("--ckpt_recon", required=True)
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)     # safer default
    ap.add_argument("--experiment_dir", default="step3_exp")
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold for defect (default 0.5)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # load reconstructor (frozen)
    ckpt = torch.load(args.ckpt_recon, map_location="cpu")
    base = ckpt.get("args",{}).get("base_channels", 32)
    netR = UNetSmall(base=base).to(device).eval()
    netR.load_state_dict(ckpt["model"])
    for p in netR.parameters(): p.requires_grad_(False)

    # datasets
    H,W = args.resize
    train_all = TrainPairs(Path(args.data_root), Path(args.step1_root), args.site, resize=(H,W))
    val_size = max(1, int(0.1*len(train_all))); train_size = len(train_all)-val_size
    train_ds, val_ds = random_split(train_all, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    test_ds  = TestPairs(Path(args.data_root), args.site, resize=(H,W))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # classifier
    clf = build_resnet18_2ch(num_classes=2).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    exp_dir = Path(args.experiment_dir); exp_dir.mkdir(parents=True, exist_ok=True)
    best = {"val_loss": math.inf}

    # train
    for epoch in range(1, args.epochs+1):
        t0=time.time()
        tr = train_epoch(train_dl, netR, clf, opt, loss_fn, device)
        vl = eval_loss(val_dl,   netR, clf, loss_fn, device)
        dt=time.time()-t0
        print(f"Epoch {epoch:02d} | train_loss={tr:.4f} | val_loss={vl:.4f} | {dt:.1f}s")

        if vl < best["val_loss"] and math.isfinite(vl):
            best = {"val_loss": vl, "epoch": epoch}
            torch.save({"clf": clf.state_dict(), "args": vars(args)}, exp_dir/"best_clf.pt")

    # safety: ensure a ckpt exists
    if not (exp_dir/"best_clf.pt").exists():
        torch.save({"clf": clf.state_dict(), "args": vars(args)}, exp_dir/"best_clf.pt")

    # test
    best_ckpt = torch.load(exp_dir/"best_clf.pt", map_location="cpu")
    clf.load_state_dict(best_ckpt["clf"])
    metrics = evaluate(test_dl, netR, clf, device, threshold=args.threshold)

    # save metrics json (without arrays)
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ("y_true","y_pred")}
    (exp_dir/"test_metrics.json").write_text(json.dumps(metrics_to_save, indent=2))
    print("Test metrics:", json.dumps(metrics_to_save, indent=2))

    # confusion matrix plot
    y_true = np.array(metrics["y_true"], dtype=int)
    y_pred = np.array(metrics["y_pred"], dtype=int)
    cm = np.zeros((2,2), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[t,p] += 1
    plot_confusion(cm, labels=("Normal","Defect"), out_path=exp_dir/"confusion_matrix.png")

    print("Saved:", (exp_dir/"best_clf.pt").resolve(), (exp_dir/"test_metrics.json").resolve(), (exp_dir/"confusion_matrix.png").resolve())

if __name__ == "__main__":
    main()
