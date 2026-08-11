import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(fp32_path: Path, out_path: Path) -> None:
    quantize_dynamic(
        str(fp32_path),
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
