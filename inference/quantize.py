import argparse
import tempfile
from pathlib import Path

import onnx
from onnx import shape_inference
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(fp32_path: Path, out_path: Path) -> None:
    model = onnx.load(str(fp32_path))
    model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / "model_clean.onnx"
        onnx.save(model, str(clean))
        quantize_dynamic(
            str(clean),
            str(out_path),
            weight_type=QuantType.QUInt8,
            per_channel=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    quantize(args.fp32, args.out)
    print(f"quantized {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
