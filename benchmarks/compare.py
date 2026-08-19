import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
import torch

from training.train import build_model
from training.data import CLASSES

MODELS = ["custom_cnn", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]


def count_params(arch: str) -> int:
    model = build_model(arch, len(CLASSES))
    return sum(p.numel() for p in model.parameters())


def run_training(config_path: Path, out_dir: Path) -> None:
    if (out_dir / "metrics.json").exists():
        print(f"=== {out_dir.name} already trained (metrics.json found) - skipping ===")
        return
    subprocess.run(
        [sys.executable, "-m", "training.train", "--config", str(config_path), "--out-dir", str(out_dir)],
        check=True,
    )


def backup_run(out_dir: Path) -> None:
    targets = [Path("/content/backup_runs")]
    drive_root = Path("/content/drive/MyDrive/checkmaize_backup")
    if Path("/content/drive/MyDrive").is_dir():
        targets.append(drive_root)
    for base in targets:
        try:
            dst = base / "runs" / out_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("best.pt", "metrics.json", "model.onnx"):
                src = out_dir / name
                if src.exists():
                    shutil.copy(src, dst / name)
        except OSError:
            pass


def build_comparison_rows(runs_dir: Path, onnx_bytes: dict[str, int]) -> list[dict]:
    rows = []
    for m in MODELS:
        with (runs_dir / m / "metrics.json").open() as f:
            metrics = json.load(f)
        rows.append(
            {
                "model": m,
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "params": count_params(m),
                "onnx_bytes": onnx_bytes.get(m, 0),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-dir", type=Path, default=Path("training/configs"))
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("benchmarks/report"))
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    onnx_bytes: dict[str, int] = {}
    if not args.skip_training:
        for m in MODELS:
            print(f"=== training {m} ===")
            run_training(args.configs_dir / f"{m}.yaml", args.runs_dir / m)
            from inference.export import export_onnx

            export_onnx(
                checkpoint=args.runs_dir / m / "best.pt",
                out_path=args.runs_dir / m / "model.onnx",
            )
            backup_run(args.runs_dir / m)
    for m in MODELS:
        p = args.runs_dir / m / "model.onnx"
        data_p = args.runs_dir / m / "model.onnx.data"
        onnx_bytes[m] = (p.stat().st_size if p.exists() else 0) + (
            data_p.stat().st_size if data_p.exists() else 0
        )
    rows = build_comparison_rows(args.runs_dir, onnx_bytes)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "params", "onnx_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    names = [r["model"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, accs)
    plt.ylim(0, 1)
    plt.ylabel("test accuracy")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(args.report_dir / "accuracy.png")
    lines = ["| model | accuracy | macro_f1 | params | onnx_bytes |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['params']} | {r['onnx_bytes']} |")
    table = "\n".join(lines)
    print(table)
    (args.report_dir / "comparison.md").write_text(table)


if __name__ == "__main__":
    main()
