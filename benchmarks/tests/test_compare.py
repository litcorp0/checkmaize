from benchmarks.compare import build_comparison_rows, MODELS

def test_model_catalog():
    assert MODELS == ["custom_cnn", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]

def test_build_rows_merges_metrics_and_sizes(tmp_path):
    import json
    out = tmp_path / "runs"
    sizes = {}
    for m in MODELS:
        d = out / m
        d.mkdir(parents=True)
        with (d / "metrics.json").open("w") as f:
            json.dump({"accuracy": 0.9, "macro_f1": 0.85}, f)
        with (d / "model.onnx").open("wb") as f:
            f.write(b"x" * 100)
        sizes[m] = 100
    rows = build_comparison_rows(out, sizes)
    assert len(rows) == len(MODELS)
    assert rows[0]["model"] == "custom_cnn"
    assert rows[0]["onnx_bytes"] == 100
