import torch
from pathlib import Path

from inference.export import export_onnx, write_contract

def test_export_produces_runnable_session(tmp_path: Path):
    import json
    from training.train import build_model
    from training.data import CLASSES
    model = build_model("custom_cnn", len(CLASSES))
    ckpt = {"state_dict": model.state_dict(), "arch": "custom_cnn", "class_names": CLASSES, "config": {}}
    torch.save(ckpt, tmp_path / "best.pt")
    out = tmp_path / "model.onnx"
    export_onnx(tmp_path / "best.pt", out)
    assert out.exists() and out.stat().st_size > 0
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = {"input": torch.rand(1, 3, 224, 224).numpy()}
    y = sess.run(None, x)
    assert y[0].shape == (1, 4)

def test_contract_doc_contains_stats(tmp_path: Path):
    from training.data import CLASSES
    doc = tmp_path / "contract.md"
    write_contract(doc, CLASSES)
    text = doc.read_text()
    assert "0.485" in text and "224" in text and "common_rust" in text
