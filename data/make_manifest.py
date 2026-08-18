import argparse
import csv
from pathlib import Path

SOURCE_FOLDER_MAP = {
    "plantvillage": {
        "Cercospora_leaf_spot Gray_leaf_spot": "gray_leaf_spot",
        "Common_rust_": "common_rust",
        "Northern_Leaf_Blight": "northern_leaf_blight",
        "healthy": "healthy",
    },
    "ccmt_ghana": {
        "Leaf blight": "northern_leaf_blight",
        "Leaf spot": "gray_leaf_spot",
        "Healthy": "healthy",
    },
}


def build_raw_manifest(raw_root: Path, out_csv: Path) -> list[dict]:
    rows: list[dict] = []
    for source, folder_map in SOURCE_FOLDER_MAP.items():
        source_dir = raw_root / source
        if not source_dir.exists():
            continue
        for folder, class_name in folder_map.items():
            class_dir = source_dir / folder
            if not class_dir.exists():
                continue
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                leaf_id = img.name.split("__")[0] if source == "plantvillage" else img.stem
                rows.append(
                    {
                        "path": str(img.relative_to(raw_root)),
                        "source": source,
                        "leaf_id": leaf_id,
                        "class": class_name,
                    }
                )
    if not rows:
        raise ValueError(f"no images found under {raw_root}; expected data/raw/<source>/<class_folder>")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/raw.csv"))
    args = parser.parse_args()
    rows = build_raw_manifest(args.raw_root, args.out)
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"wrote {len(rows)} rows to {args.out}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
