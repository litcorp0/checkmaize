import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from training.data import CLASSES, ManifestDataset, MEAN, STD
from training.train import build_model


def _top1(sess, loader) -> float:
    preds, labels = [], []
    for x, y in loader:
        logits = sess.run(None, {"input": x.numpy()})[0]
        preds.extend(logits.argmax(axis=1).tolist())
        labels.extend(y.tolist())
    return accuracy_score(labels, preds)


def verify(checkpoint, fp32_path, int8_path, test_manifest, data_root) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_ds = ManifestDataset(test_manifest, Path(data_root), limit=64)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    torch_logits, onnx_logits = [], []
    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for x, _ in loader:
            torch_logits.append(model(x).numpy())
            onnx_logits.append(fp32_sess.run(None, {"input": x.numpy()})[0])
    torch_logits = np.concatenate(torch_logits)
    onnx_logits = np.concatenate(onnx_logits)
    parity_max_diff = float(np.abs(torch_logits - onnx_logits).max())
    full_ds = ManifestDataset(test_manifest, Path(data_root))
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)
    fp32_acc = _top1(fp32_sess, full_loader)
    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8_acc = _top1(int8_sess, full_loader)
    delta = int8_acc - fp32_acc
    report = {
        "parity_max_diff": parity_max_diff,
        "parity_pass": parity_max_diff < 1e-3,
        "fp32_accuracy": fp32_acc,
        "int8_accuracy": int8_acc,
        "delta": delta,
        "ship_int8": delta >= -0.01,
    }
    return report


def generate_fixture(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0, 255, 32, dtype=np.uint8)
    x = np.linspace(255, 0, 32, dtype=np.uint8)
    r = np.broadcast_to(y[None, :], (32, 32))
    g = np.broadcast_to(x[:, None], (32, 32))
    b = (r.astype(np.uint16) + g.astype(np.uint16)) // 2
    img = np.stack([r, g, b.astype(np.uint8)], axis=-1)
    png = out_dir / "sample.png"
    Image.fromarray(img).save(png)
    rgb = img.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    norm = (chw - np.array(MEAN).reshape(3, 1, 1)) / np.array(STD).reshape(3, 1, 1)
    tensor_path = out_dir / "reference_tensor.json"
    tensor_path.write_text(json.dumps(norm.flatten().tolist()))
    return png, tensor_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--report-out", type=Path, default=Path("inference/verify_report.json"))
    parser.add_argument("--fixture-out", type=Path, default=Path("app/src/ml/__tests__/fixtures"))
    args = parser.parse_args()
    report = verify(args.checkpoint, args.fp32, args.int8, args.test_manifest, args.data_root)
    args.report_out.write_text(json.dumps(report, indent=2))
    generate_fixture(args.fixture_out)
    print(json.dumps(report, indent=2))
    if not report["parity_pass"]:
        raise SystemExit("FAIL: fp32 export parity exceeds tolerance")
    if not report["ship_int8"]:
        print("WARNING: int8 accuracy drop > 1%; ship fp32 instead")


if __name__ == "__main__":
    main()
