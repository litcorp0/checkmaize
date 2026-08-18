import csv
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from data.make_manifest import build_raw_manifest


def _make_tree(root: Path) -> None:
    pv = root / "plantvillage" / "Common_rust_"
    pv.mkdir(parents=True)
    for name in ("leafA__1.jpg", "leafA__2.jpg"):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(pv / name)
    g = root / "ccmt_ghana" / "Leaf blight"
    g.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(g / "blight_01.jpg")


def test_build_raw_manifest_maps_and_groups():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root)
        out = root / "raw.csv"
        rows = build_raw_manifest(root, out)
        assert len(rows) == 3
        by_class = {r["class"] for r in rows}
        assert by_class == {"common_rust", "northern_leaf_blight"}
        cr = [r for r in rows if r["class"] == "common_rust"]
        assert {r["leaf_id"] for r in cr} == {"leafA"}
        assert all(r["path"].endswith(".jpg") for r in rows)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert [r["class"] for r in reader] == [r["class"] for r in rows]
