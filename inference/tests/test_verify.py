import json
import numpy as np
import torch
from pathlib import Path

from inference.verify import generate_fixture

def test_fixture_roundtrip(tmp_path: Path):
    png, tensor_path = generate_fixture(tmp_path)
    assert png.exists() and tensor_path.exists()
    ref = json.loads(tensor_path.read_text())
    assert len(ref) == 32 * 32 * 3
    from PIL import Image
    img = np.array(Image.open(png).convert("RGB")).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)).flatten()
    img = (img - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
    assert np.allclose(img.flatten(), np.array(ref), atol=1e-5)
