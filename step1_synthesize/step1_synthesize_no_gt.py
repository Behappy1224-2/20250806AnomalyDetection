# step1_synthesize_no_gt.py
# Micro-defects + mask outputs (originals/images/masks)
import argparse, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def list_images(folder: Path):
    files = []
    for e in IMG_EXTS: files += list(folder.glob(f"*{e}"))
    return sorted(files)

def blend_on(base: Image.Image, overlay: Image.Image, alpha: float):
    return Image.blend(base, overlay, alpha)

# ---------- Drawing ops that ALSO paint a mask ----------
def draw_hairline(img: Image.Image, msk: Image.Image, n_lines=1):
    """very thin scratches (1-2 px), faint; paints white strokes on mask"""
    im = img.copy(); mm = msk.copy()
    d = ImageDraw.Draw(im); dm = ImageDraw.Draw(mm)
    W,H = im.size
    for _ in range(n_lines):
        x1,y1 = random.randint(0,W-1), random.randint(0,H-1)
        angle = random.uniform(0, math.pi)
        L = random.randint(int(0.2*min(W,H)), int(0.6*min(W,H)))
        x2 = int(x1 + L*math.cos(angle)); y2 = int(y1 + L*math.sin(angle))
        color = random.randint(140,180) if random.random()<0.5 else random.randint(60,100)
        w = random.choice([1,2])
        d.line([(x1,y1),(x2,y2)], fill=(color, color, color), width=w)
        dm.line([(x1,y1),(x2,y2)], fill=255, width=max(1,w))  # paint mask
    im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3,1.0)))
    mm = mm.filter(ImageFilter.GaussianBlur(radius=0.6))
    return im, mm

def draw_pin_dots(img: Image.Image, msk: Image.Image, n_dots=3):
    """tiny dots (r=1-3 px); paints filled circles on mask"""
    im = img.copy(); mm = msk.copy()
    d = ImageDraw.Draw(im); dm = ImageDraw.Draw(mm)
    W,H = im.size
    for _ in range(n_dots):
        r = random.randint(1,3)
        cx, cy = random.randint(r, W-r-1), random.randint(r, H-r-1)
        c = random.randint(30,70) if random.random()<0.7 else random.randint(180,220)
        bbox = (cx-r, cy-r, cx+r, cy+r)
        d.ellipse(bbox, fill=(c,c,c))
        dm.ellipse(bbox, fill=255)
    im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2,0.8)))
    mm = mm.filter(ImageFilter.GaussianBlur(radius=0.6))
    return im, mm

def draw_faint_streak(img: Image.Image, msk: Image.Image):
    """narrow low-contrast streak (vertical or horizontal); paints bar on mask"""
    im = img.copy(); mm = msk.copy()
    d = ImageDraw.Draw(im); dm = ImageDraw.Draw(mm)
    W,H = im.size
    vertical = random.random()<0.5
    if vertical:
        x = random.randint(0, W-1)
        w = random.choice([1,2,3])
        c = random.randint(170,200) if random.random()<0.5 else random.randint(55,85)
        rect = (x, 0, x+w, H)
        d.rectangle(rect, fill=(c,c,c))
        dm.rectangle(rect, fill=255)
    else:
        y = random.randint(0, H-1)
        h = random.choice([1,2,3])
        c = random.randint(170,200) if random.random()<0.5 else random.randint(55,85)
        rect = (0, y, W, y+h)
        d.rectangle(rect, fill=(c,c,c))
        dm.rectangle(rect, fill=255)
    im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3,1.2)))
    mm = mm.filter(ImageFilter.GaussianBlur(radius=0.6))
    return im, mm

def synth_micro_with_mask(base: Image.Image):
    """
    Compose 1-3 micro defects with tiny alpha, and build a binary mask showing coverage.
    Returns: (synthetic_image_PIL, mask_PIL mode "L")
    """
    W,H = base.size
    # start with identity overlays
    over = base.copy()
    mask = Image.new("L", (W,H), 0)

    # choose ops (guarantee at least one)
    ops = []
    if random.random() < 0.9:
        ops.append(lambda im,mm: draw_hairline(im, mm, n_lines=random.choice([1,1,2])))
    if random.random() < 0.9:
        ops.append(lambda im,mm: draw_pin_dots(im, mm, n_dots=random.choice([1,2,3,4])))
    if random.random() < 0.7:
        ops.append(lambda im,mm: draw_faint_streak(im, mm))
    random.shuffle(ops)
    if len(ops) == 0:
        ops = [lambda im,mm: draw_pin_dots(im, mm, n_dots=random.choice([2,3]))]

    # apply 1..3 ops onto over & mask
    k = random.randint(1, min(3, len(ops)))
    for f in ops[:k]:
        over, mask = f(over, mask)

    # very low alpha to keep it subtle
    alpha = random.uniform(0.15, 0.25)
    out = blend_on(base, over, alpha)

    # optional light blur for realism
    if random.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.3))

    # binarize/clip mask to {0,255}
    m_np = np.array(mask, dtype=np.uint8)
    m_np = (m_np > 10).astype(np.uint8) * 255
    mask = Image.fromarray(m_np, mode="L")
    return out, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dataset root containing SP3/SP5")
    ap.add_argument("--site", choices=["SP3","SP5"], required=True)
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256], help="H W")
    ap.add_argument("--n_per_image", type=int, default=2, help="how many micro synthetics per normal image")
    ap.add_argument("--out_root", required=True, help="step1_outputs root (existing one is fine)")
    args = ap.parse_args()

    site_root = Path(args.data_root)/args.site
    src_dir = site_root/"train"/"defect-free"
    imgs = []
    for e in IMG_EXTS: imgs += list((src_dir).glob(f"*{e}"))
    imgs = sorted(imgs)
    if len(imgs)==0:
        raise SystemExit("No train/defect-free images found.")

    out_site = Path(args.out_root)/args.site
    dir_img = out_site/"images"; dir_org = out_site/"originals"; dir_msk = out_site/"masks"
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_org.mkdir(parents=True, exist_ok=True)
    dir_msk.mkdir(parents=True, exist_ok=True)

    H,W = args.resize
    counter = 0
    for p in imgs:
        orig = Image.open(p).convert("RGB").resize((W,H), Image.BILINEAR)
        stem = p.stem

        # save original once per source
        (dir_org/f"{stem}_src.png").parent.mkdir(parents=True, exist_ok=True)
        orig.save(dir_org/f"{stem}_src.png")

        for _ in range(args.n_per_image):
            syn, m = synth_micro_with_mask(orig)
            syn.save(dir_img/f"{stem}_sM{counter:05d}.png")
            m.save(dir_msk/f"{stem}_sM{counter:05d}_mask.png")
            counter += 1

    print(f"Done. Wrote {counter} micro synthetics with masks to {out_site}.")

if __name__ == "__main__":
    main()
