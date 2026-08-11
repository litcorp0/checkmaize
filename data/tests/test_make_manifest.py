import csv
from pathlib import Path
import tempfile

from data.make_manifest import build_raw_manifest


def _make_tree(root: Path) -> None:
    pv = root / "plantvillage" / "Common_rust_"
    pv.mkdir(parents=True)
    (pv / "leafA__1.jpg").touch()
    (pv / "leafA__2.jpg").touch()
    g = root / "ccmt_ghana" / "Leaf blight"
    g.mkdir(parents=True)
    (g / "blight_01.jpg").touch()


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
