# CheckMaize — Smart Maize Leaf Disease Detection (Ghana)

On-device maize leaf disease classification for Ghanaian farming conditions.
Research pipeline (PyTorch on Colab) + Expo SDK 57 mobile app running int8 ONNX.

## Repository layout

- `data/` — dataset ingestion + leakage-safe split manifests
- `training/` — PyTorch fine-tuning drivers + configs
- `benchmarks/` — RQ3 architecture comparison + report artifacts
- `inference/` — ONNX export, int8 quantization, parity verification
- `app/` — Expo React Native application
- `docs/` — design spec, dataset notes, ONNX contract, pilot protocol

## Workflow (Colab)

1. `colab/01_dataset.ipynb` — build dataset + splits
2. `colab/02_train_benchmark.ipynb` — train 5 candidate models, compare
3. `colab/03_export.ipynb` — export/quantize/verify the winner, produce app artifacts

Detailed notebook instructions live in `docs/colab-workflow.md`.
