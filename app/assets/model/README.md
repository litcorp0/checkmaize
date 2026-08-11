This directory holds the shipped model file produced by the Colab notebook
`colab/03_export.ipynb`. After running the notebook and downloading
`artifacts.zip`, copy these three files into this directory:

- `model_int8.onnx` (the quantized ONNX model, ~3-5 MB)
- `labels.json` (class index: ["common_rust","gray_leaf_spot","northern_leaf_blight","healthy"])
- `metrics.json` (model performance stats)

These files must be committed for EAS Build to include them in the APK/IPA.
