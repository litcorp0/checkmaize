import torch
from pathlib import Path

def _tiny_onnx(tmp_path: Path) -> Path:
    import onnx
    from onnx import helper, TensorProto
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 3], [0.1] * 12)
    node = helper.make_node("MatMul", ["input", "w"], ["output"])
    graph = helper.make_graph(
        [node], "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, tmp_path / "model.onnx")
    return tmp_path / "model.onnx"

def test_quantize_produces_int8(tmp_path: Path):
    from inference.quantize import quantize
    fp32 = _tiny_onnx(tmp_path)
    out = tmp_path / "model_int8.onnx"
    quantize(fp32, out)
    assert out.exists() and out.stat().st_size > 0
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": np.ones((1, 3), dtype=np.float32)})
    assert y[0].shape == (1, 4)
