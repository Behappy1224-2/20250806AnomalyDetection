
import argparse
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS])

def mask_blobs(size: Tuple[int,int], num=(1,5), scale=(0.05,0.20)) -> Image.Image:
    W,H = size
    m = Image.new("L", (W,H), 0) #full black mask
    dr = ImageDraw.Draw(m)
    for _ in range(random.randint(*num)):
        sx = int(random.uniform(*scale)*W) #eclipse x-axis 橢圓x軸長度
        sy = int(random.uniform(*scale)*H) #eclipse y-axis
        cx = random.randint(0, W-1); cy = random.randint(0, H-1) #eclipse center point(x,y)
        dr.ellipse([cx-sx//2, cy-sy//2, cx+sx//2, cy+sy//2], fill=255) #draw eclipse with white
    m = m.filter(ImageFilter.GaussianBlur(radius=1))
    return m

def mask_scratches(size: Tuple[int,int], num=(1,4), width=(1,5)) -> Image.Image:
    W,H = size
    m = Image.new("L", (W,H), 0)
    dr = ImageDraw.Draw(m)
    for _ in range(random.randint(*num)):
        x0 = random.randint(0, W-1); y0 = random.randint(0, H-1)
        x1 = random.randint(0, W-1); y1 = random.randint(0, H-1)
        w  = random.randint(*width)
        dr.line([x0,y0,x1,y1], fill=255, width=w)
    m = m.filter(ImageFilter.GaussianBlur(radius=0.8))
    return m

def mask_rect_cutout(size: Tuple[int,int], num=(0,2), scale=(0.05,0.2)) -> Image.Image:
    W,H = size
    m = Image.new("L", (W,H), 0)
    dr = ImageDraw.Draw(m)
    for _ in range(random.randint(*num)):
        sx = int(random.uniform(*scale)*W); sy = int(random.uniform(*scale)*H)
        x0 = random.randint(0, max(0, W-sx)); y0 = random.randint(0, max(0, H-sy))
        dr.rectangle([x0,y0,x0+sx,y0+sy], fill=255)
    return m

def apply_in_mask(img: Image.Image, mask: Image.Image, severity: float=0.6) -> Image.Image:
    img = img.convert("RGB")
    base = np.array(img).astype(np.float32)

    # print(np.array(mask.resize(img.size, Image.NEAREST)).shape)
    # for i in (np.array(mask.resize(img.size, Image.NEAREST))):
    #     print(i)
    m = (np.array(mask.resize(img.size, Image.NEAREST)) > 127) 
    # for i in m:
    #     print(i)
    modes_all = ["darken","noise","desaturate","brighten","blur"]
    k = random.choice([1,2,3])
    modes = random.sample(modes_all, k=k)

    out = base.copy()
    region = m

    if "darken" in modes:
        out[region] *= (1.0 - 0.55*severity)
    if "brighten" in modes:
        out[region] += 50*severity
    if "noise" in modes:
        out[region] += np.random.normal(0, 25*severity, size=out[region].shape)
    if "desaturate" in modes:
        w = np.array([0.299,0.587,0.114], dtype=np.float32)
        gray = (out[region]*w).sum(axis=1, keepdims=True)
        mix = 0.6 + 0.4*severity
        out[region] = mix*gray + (1-mix)*out[region]
    out = np.clip(out, 0, 255)

    if "blur" in modes:
        pil_blur = Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(1, int(2*severity))))
        blur_np = np.array(pil_blur).astype(np.float32)
        out[region] = blur_np[region]

    return Image.fromarray(out.astype(np.uint8))

def save_grid(triplets, out_png):
    cols=3; rows=len(triplets)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.8, rows*3.2))
    if rows==1:
        axes = np.array([axes])
    for r,(orig,mask,synt) in enumerate(triplets):
        axes[r,0].imshow(orig); axes[r,0].set_axis_off(); axes[r,0].set_title("original")
        axes[r,1].imshow(mask, cmap="gray"); axes[r,1].set_axis_off(); axes[r,1].set_title("mask")
        axes[r,2].imshow(synt); axes[r,2].set_axis_off(); axes[r,2].set_title("synthetic")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Step 1: Create synthetic anomaly images (NO GroundTruth)")
    ap.add_argument("--data_root", required=True, help="Path containing SP3/SP5")
    ap.add_argument("--site", choices=["SP3","SP5"], default="SP3")
    ap.add_argument("--out_dir", default="synthetic_data")
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--num", type=int, required=True, help="How many normals image to Synthesize?")
    ap.add_argument("--variants", type=int, default=1, help="Synthetic variants per base image")
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); 
    np.random.seed(args.seed)

    site_root = Path(args.data_root) / args.site
    train_normals = site_root/'train'/'defect-free'
    normals = list_images(train_normals)
    print(len(normals))
    if len(normals)==0:
        print('No normal images under:', train_normals); return

    H,W = args.resize
    out_base = Path(args.out_dir)/args.site
    out_img = out_base/'images'
    out_mask = out_base/'masks'
    out_src  = out_base/'originals'
    for d in [out_img,out_mask,out_src]:
        d.mkdir(parents=True, exist_ok=True)

    random.shuffle(normals)
    normals = normals[:args.num]

    preview = []
    counter=0
    for p in normals:
        try:
            base = Image.open(p).convert('RGB').resize((W,H), Image.BILINEAR)
        except Exception as e:
            print('Skip unreadable:', p, e); continue

        for v in range(args.variants):
            m = mask_blobs((W,H))
            if random.random()<0.6:
                m = ImageChops.lighter(m, mask_scratches((W,H)))
            if random.random()<0.3:
                m = ImageChops.lighter(m, mask_rect_cutout((W,H)))

            synt = apply_in_mask(base, m, severity=args.severity)

            stem = Path(p).stem
            base.save(out_src/f"{stem}_src.png")
            synt.save(out_img/f"{stem}_s{counter:04d}.png")
            m.save(out_mask/f"{stem}_s{counter:04d}_mask.png")

            # if len(preview)<8:
            #     preview.append((base, m, synt))

            counter += 1

    # grid = out_base/'step1_preview.png'
    # if preview:
    #     save_grid(preview, grid)
    #     print('Preview saved:', grid.resolve())

    print('Done. Wrote:', counter, 'synthetic images to', out_img.resolve())

if __name__ == '__main__':
    main()