This directory holds the shipped model files produced by the Colab notebook
`colab/03_export.ipynb`. After running the notebook and downloading
`artifacts.zip`, copy these three files into this directory:

- `model.onnx` (the shipped ONNX model — fp32, ~6 MB; the int8 variant is used only if the verify step's `ship_int8` flag is true)
- `labels.json` (class index: ["common_rust","gray_leaf_spot","northern_leaf_blight","healthy"])
- `metrics.json` (model performance stats)

These files must be committed for EAS Build to include them in the APK/IPA.
