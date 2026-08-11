import csv
from pathlib import Path
import tempfile

from data.make_splits import build_splits

def _raw_rows(tmp: Path) -> None:
    rows = []
    for i in range(40):
        rows.append({"path": f"ccmt_ghana/Leaf blight/{i}.jpg", "source": "ccmt_ghana", "leaf_id": f"g{i}", "class": "northern_leaf_blight"})
    for i in range(40):
        rows.append({"path": f"ccmt_ghana/Healthy/{i}.jpg", "source": "ccmt_ghana", "leaf_id": f"h{i}", "class": "healthy"})
    for i in range(40):
        rows.append({"path": f"ccmt_ghana/Leaf spot/{i}.jpg", "source": "ccmt_ghana", "leaf_id": f"s{i}", "class": "gray_leaf_spot"})
    for i in range(120):
        rows.append({"path": f"plantvillage/Common_rust_/{i}.jpg", "source": "plantvillage", "leaf_id": f"c{i}", "class": "common_rust"})
    with (tmp / "raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        writer.writeheader()
        writer.writerows(rows)

def _read(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))

def test_splits_respect_leakage_and_axes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _raw_rows(tmp)
        build_splits(tmp / "raw.csv", tmp, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
        train = _read(tmp / "train.csv")
        val = _read(tmp / "val.csv")
        test = _read(tmp / "test.csv")
        d_train = _read(tmp / "domain_shift_train.csv")
        d_test = _read(tmp / "domain_shift_test.csv")
        assert test, "test.csv must not be empty"
        assert all(r["source"] == "ccmt_ghana" for r in test), "main test must be Ghana-only"
        train_ids = {(r["source"], r["leaf_id"]) for r in train}
        test_ids = {(r["source"], r["leaf_id"]) for r in test}
        assert not (train_ids & test_ids), "leaf leakage between train and test"
        assert all(r["source"] == "plantvillage" for r in d_train), "domain-shift train is PlantVillage-only"
        ghana_total = sum(1 for r in train + val + test if r["source"] == "ccmt_ghana")
        assert ghana_total == 120
        assert len(d_test) == 120, "domain-shift test = all Ghana images"
        classes_in_test = {r["class"] for r in test}
        assert classes_in_test == {"northern_leaf_blight", "healthy", "gray_leaf_spot"}

def test_splits_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _raw_rows(tmp)
        build_splits(tmp / "raw.csv", tmp, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
        first = (tmp / "test.csv").read_bytes()
        build_splits(tmp / "raw.csv", tmp, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
        assert (tmp / "test.csv").read_bytes() == first
