# step2_eval_recon.py
# Evaluate the trained U-Net on real SPx test images.
# It measures per-image mean residual: mean(|x - recon(x)|),
# reports group means (normal vs defect), and saves a few panels.

import argparse, json, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def list_images(folder: Path) -> List[Path]:
    files = []
    for ext in IMG_EXTS:
        files += list(folder.glob(f"*{ext}"))
    return sorted(files)

# ---- same compact U-Net used in training ----
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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

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
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x  = self.up1(x3, x2); x = self.up2(x, x1); x = self.outc(x)
        return torch.sigmoid(x)

def load_image(p: Path, size: Tuple[int,int]) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    return TF.to_tensor(img)  # [0,1] CxHxW

def save_panel(inp, recon, out_path: Path):
    to_pil = TF.to_pil_image
    inp_p, rec_p = to_pil(inp.clamp(0,1)), to_pil(recon.clamp(0,1))
    w,h = inp_p.size; panel = Image.new("RGB",(w*2,h))
    panel.paste(inp_p,(0,0)); panel.paste(rec_p,(w,0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Evaluate reconstructor on real test images")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt from Step 2 training")
    ap.add_argument("--data_root", required=True, help="dataset root with SP3/SP5")
    ap.add_argument("--site", choices=["SP3","SP5"], default="SP3")
    ap.add_argument("--resize", type=int, nargs=2, default=[128,128], help="H W")
    ap.add_argument("--out_dir", default="step2_eval")
    ap.add_argument("--num_show", type=int, default=6)
    args = ap.parse_args()

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    base = ckpt.get("args",{}).get("base_channels", 32)  # fallback
    net = UNetSmall(base=base).eval()
    net.load_state_dict(ckpt["model"])

    H,W = args.resize
    site = Path(args.data_root)/args.site
    d_norm = site/"test"/"defect-free"
    d_def  = site/"test"/"defect"
    normals = list_images(d_norm)
    defects = list_images(d_def)
    if len(normals)==0 or len(defects)==0:
        raise SystemExit("No test images found. Check paths.")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def batch_eval(paths: List[Path], label: str):
        res_vals = []
        to_show = []
        for i,p in enumerate(paths):
            x = load_image(p, (H,W)).unsqueeze(0)        # 1xCxHxW
            with torch.no_grad():
                y = net(x)
            residual = (x - y).abs().mean().item()       # mean absolute residual
            res_vals.append(residual)
            if len(to_show) < args.num_show:
                to_show.append((x.squeeze(0), y.squeeze(0), p.name))
        # save a few panels
        for i,(xi,yi,name) in enumerate(to_show):
            save_panel(xi, yi, out_dir / f"{label}_panel_{i:02d}_{name}")
        return np.array(res_vals, dtype=np.float32)

    t0 = time.time()
    r_norm = batch_eval(normals, "normal")
    r_def  = batch_eval(defects, "defect")
    dt = time.time() - t0

    stats = {
        "normal_mean": float(r_norm.mean()), "normal_std": float(r_norm.std()),
        "defect_mean": float(r_def.mean()),  "defect_std": float(r_def.std()),
        "n_normal": int(len(r_norm)), "n_defect": int(len(r_def)),
        "seconds": dt, "resize": [H,W]
    }
    (out_dir/"residual_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # quick histogram
    plt.figure(figsize=(6,4))
    plt.hist(r_norm, bins=30, alpha=0.6, label="normal")
    plt.hist(r_def,  bins=30, alpha=0.6, label="defect")
    plt.xlabel("mean |x - recon(x)|"); plt.ylabel("#images"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"residual_hist.png", dpi=140)
    plt.close()

if __name__ == "__main__":
    main()
