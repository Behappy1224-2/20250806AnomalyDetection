# step2_train_unet.py
# Single-file Step 2: train a compact U-Net to reconstruct clean images from synthetic defects.
# Input  : step1 outputs (images/)
# Target : step1 outputs (originals/)
import os, torch
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import argparse
from pathlib import Path
import time
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ----------------------------
# Dataset that reads Step-1 outputs
# ----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _list_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files += list(folder.glob(f"*{ext}"))
    return sorted(files)

class Step1Pairs(Dataset):
    # """
    # Pairs synthetic images (step1_dir/site/.../images/*.png)
    # with clean originals (step1_dir/site/.../originals/*_src.png).
    # Matching is by filename stem before the synthetic suffix `_sXXXX`.
    # """
    def __init__(self, step1_site_dir: Path, resize=(128,128)):
        self.step1_site_dir = Path(step1_site_dir)
        self.dir_images = self.step1_site_dir / "images"
        self.dir_originals = self.step1_site_dir / "originals"
        if not self.dir_images.exists() or not self.dir_originals.exists():
            raise FileNotFoundError(f"Expected folders not found:\n  {self.dir_images}\n  {self.dir_originals}")

        self.resize = resize
        self.items = []  # list of tuples (synthetic_path, original_path)

        syn_files = _list_images(self.dir_images)
        # Build index of originals by stem (without trailing _src)
        orig_index = {}
        for p in _list_images(self.dir_originals):
            st = p.stem  # e.g., foo_src
            if st.endswith("_src"):
                orig_index[st[:-4]] = p  # map 'foo' -> .../foo_src.png

        for sp in syn_files:
            st = sp.stem      # e.g., foo_s0001
            base = st.split("_s")[0]  # 'foo'
            if base in orig_index:
                self.items.append((sp, orig_index[base]))

        if len(self.items) == 0:
            raise RuntimeError("No synthetic-original pairs matched. "
                               "Check your step1 outputs structure and names.")

    def __len__(self):
        return len(self.items)

    def _load_rgb(self, p: Path):
        img = Image.open(p).convert("RGB")
        if self.resize:
            W, H = self.resize[1], self.resize[0]  # resize=(H, W)
            img = img.resize((W, H), Image.BILINEAR)
        # to tensor [0,1]
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        syn_p, orig_p = self.items[idx]
        x_in = self._load_rgb(syn_p)   # synthetic
        x_tg = self._load_rgb(orig_p)  # clean target
        return {"input": x_in, "target": x_tg, "syn_path": str(syn_p), "orig_path": str(orig_p)}


# ----------------------------
# Compact U-Net (very small, CPU-friendly)
# ----------------------------
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
        # pad if odd sizes
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x  = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.up1   = Up(base*4 + base*2, base*2)
        self.up2   = Up(base*2 + base, base)
        self.outc  = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x  = self.up1(x3, x2)
        x  = self.up2(x, x1)
        x  = self.outc(x)
        return torch.sigmoid(x)


# ----------------------------
# Training
# ----------------------------
def save_samples(net, batch, out_dir: Path, max_save=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    net.eval()
    x = batch["input"]
    y = batch["target"]
    with torch.no_grad():
        yhat = net(x)
    # Save a few recon panels: input | recon | target horizontally
    to_pil = TF.to_pil_image
    n = min(x.size(0), max_save)
    for i in range(n):
        inp   = to_pil(x[i].clamp(0,1))
        recon = to_pil(yhat[i].clamp(0,1))
        targ  = to_pil(y[i].clamp(0,1))
        # simple horizontal concat
        w, h = inp.size
        panel = Image.new("RGB", (w*3, h))
        panel.paste(inp,   (0,   0))
        panel.paste(recon, (w,   0))
        panel.paste(targ,  (w*2, 0))
        panel.save(out_dir / f"sample_{i:02d}.png")

def main():
    ap = argparse.ArgumentParser(description="Step 2: Train U-Net reconstructor on Step-1 synthetic pairs")
    ap.add_argument("--step1_site_dir", type=str, required=False,
                    help="Path to Step-1 site's output folder that contains images/ and originals/")
    ap.add_argument("--step1_root", type=str, required=False,
                    help="If set, auto-pick site subfolder inside this root (e.g., step1_outputs/SP3)")
    ap.add_argument("--site", type=str, default="SP3", choices=["SP3","SP5"],
                    help="Used only if --step1_root is given")
    ap.add_argument("--resize", type=int, nargs=2, default=[128,128], help="H W")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--experiment_dir", type=str, default="step2_results")
    args = ap.parse_args()

    # Resolve Step-1 site dir
    if args.step1_site_dir:
        site_dir = Path(args.step1_site_dir)
    elif args.step1_root:
        site_dir = Path(args.step1_root) / args.site
    else:
        raise SystemExit("Provide either --step1_site_dir or (--step1_root and --site).")

    ds = Step1Pairs(site_dir, resize=tuple(args.resize))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

    device = torch.device("cpu")  # your machine; no CUDA
    net = UNetSmall(in_ch=3, out_ch=3, base=args.base_channels).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    l1  = nn.L1Loss()

    exp_dir = Path(args.experiment_dir)
    ckpt_dir = exp_dir / "checkpoints"
    samp_dir = exp_dir / "samples"
    exp_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(exist_ok=True); samp_dir.mkdir(exist_ok=True)

    print(f"Dataset size: {len(ds)} pairs  |  Image size: {args.resize[0]}x{args.resize[1]}")
    print(f"Training on CPU: epochs={args.epochs}, bs={args.batch_size}, base_ch={args.base_channels}")

    for epoch in range(1, args.epochs+1):
        net.train()
        t0 = time.time()
        tot = 0.0; n = 0
        for batch in dl:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            yhat = net(x)
            loss = l1(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        dt = time.time() - t0
        avg = tot / max(1, n)
        print(f"Epoch {epoch:02d} | L1={avg:.4f} | {dt:.1f}s")

        # save ckpt + a few samples
        torch.save({"model": net.state_dict(),
                    "args": vars(args)}, ckpt_dir / f"unet_small_e{epoch}.pt")
        # grab a small fixed batch for preview
        save_samples(net, next(iter(dl)), samp_dir / f"epoch_{epoch:02d}")

    print("Done. Check:", exp_dir.resolve())

if __name__ == "__main__":
    # Reproducibility
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    main()
