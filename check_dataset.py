from pathlib import Path

def count_images(folder: Path, exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}):
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)

def check_site(root: Path, site: str):
    base = root / site
    paths = {
        "gt_masks": base / "GroundTruth" / "defect",
        "test_defect": base / "test" / "defect",
        "test_normal": base / "test" / "defect-free",
        "train_normal": base / "train" / "defect-free",
    }
    print(f"\n== {site} ==")
    for k, p in paths.items():
        if not p.exists():
            print(f"  MISSING: {k} -> {p}")
        else:
            print(f"  {k:12s}: {count_images(p)} files ({p})")

if __name__ == "__main__":
    # CHANGE THIS to your dataset root that contains SP3 and/or SP5
    data_root = Path(r"D:/Lab Summer Course/2025_0806_Anomaly_Detection/20250806Dataset")
    for site in ["SP3", "SP5"]:
        if (data_root / site).exists():
            check_site(data_root, site)
