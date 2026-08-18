import csv
from pathlib import Path
import tempfile
import numpy as np
import torch
from PIL import Image

from training.data import ManifestDataset, CLASSES, MEAN, STD, eval_transforms

def _make_csv(tmp: Path, n: int = 8) -> Path:
    raw = tmp / "img"
    raw.mkdir()
    rows = []
    for i in range(n):
        cls = CLASSES[i % len(CLASSES)]
        folder = raw / cls
        folder.mkdir(exist_ok=True)
        img = Image.fromarray(np.full((240, 320, 3), 120 + i, dtype=np.uint8))
        p = folder / f"{i}.jpg"
        img.save(p)
        rows.append({"path": f"img/{cls}/{i}.jpg", "source": "ccmt_ghana", "leaf_id": str(i), "class": cls})
    csv_path = tmp / "rows.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        w.writeheader()
        w.writerows(rows)
    return csv_path

def test_dataset_and_contract():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv_path = _make_csv(tmp)
        ds = ManifestDataset(csv_path, root=tmp)
        assert len(ds) == 8
        x, y = ds[0]
        assert tuple(x.shape) == (3, 224, 224)
        assert x.dtype == torch.float32
        assert 0 <= y < len(CLASSES)

def test_eval_transforms_match_contract():
    img = Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8))
    x = eval_transforms()(img)
    assert tuple(x.shape) == (3, 224, 224)
    means = torch.tensor(MEAN).view(3, 1, 1)
    stds = torch.tensor(STD).view(3, 1, 1)
    expected = (128 / 255 - means) / stds
    assert torch.allclose(x, expected.expand_as(x), atol=1e-5)
