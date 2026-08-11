import json
import tempfile
from pathlib import Path
import yaml

from training.train import main

def test_train_smoke(tmp_path: Path):
    cfg = {
        "arch": "custom_cnn",
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "data_root": str(tmp_path),
        "seed": 42,
        "train_manifest": "train.csv",
        "val_manifest": "val.csv",
        "test_manifest": "test.csv",
    }
    from data.make_manifest import build_raw_manifest
    from data.make_splits import build_splits
    import numpy as np
    from PIL import Image
    raw = tmp_path / "raw"
    (raw / "plantvillage" / "Common_rust_").mkdir(parents=True)
    (raw / "ccmt_ghana" / "Leaf blight").mkdir(parents=True)
    (raw / "ccmt_ghana" / "Leaf spot").mkdir(parents=True)
    (raw / "ccmt_ghana" / "Healthy").mkdir(parents=True)
    for i in range(30):
        img = Image.fromarray(np.full((224, 224, 3), 100 + i, dtype=np.uint8))
        img.save(raw / "plantvillage" / "Common_rust_" / f"c{i}.jpg")
        img.save(raw / "ccmt_ghana" / "Leaf blight" / f"g{i}.jpg")
        img.save(raw / "ccmt_ghana" / "Leaf spot" / f"g2_{i}.jpg")
        img.save(raw / "ccmt_ghana" / "Healthy" / f"g3_{i}.jpg")
    build_raw_manifest(raw, tmp_path / "raw.csv")
    build_splits(tmp_path / "raw.csv", tmp_path, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w") as f:
        yaml.dump(cfg, f)
    out = tmp_path / "out"
    main(["--config", str(cfg_path), "--out-dir", str(out)])
    assert (out / "best.pt").exists()
    with (out / "metrics.json").open() as f:
        metrics = json.load(f)
    assert "accuracy" in metrics and "macro_f1" in metrics
