# Colab Workflow

All Python runs on Google Colab (free). The repo is the source of truth; notebooks are drivers.

## Prerequisites per session

- Push the repo to GitHub (or upload a zip to `/content/checkmaize.zip`).
- Notebook 01: download PlantVillage via HF (auto), upload the CCMT Ghana zip
  (`crop-pest-and-disease-detection.zip` from the Kaggle mirror, or
  `Raw Data.zip` from Mendeley DOI 10.17632/bwh3zbpkpv.1) via the files pane.
  The notebook accepts both layouts automatically.
- Notebook 03: set the `winner` variable to the model chosen from the Task 8 table.

## Notebooks

1. `colab/01_dataset.ipynb` - extracts PlantVillage + CCMT, builds `raw.csv`, splits, tests.
   Download `splits.zip` -> local `data/manifests/`, commit.
2. `colab/02_train_benchmark.ipynb` - smoke tests, then full 5-model benchmark (~2-3 h).
   Download `report.zip` -> `benchmarks/report/`, commit. Pick winner.
3. `colab/03_export.ipynb` - export, int8 quantize, verify, app fixtures + artifacts.
   Download `artifacts.zip` -> `artifacts/`, `fixtures.zip` -> `app/src/ml/__tests__/fixtures/`,
   and `onnx-contract.md` -> `docs/`. Commit (model goes to `app/assets/model/` in Task 20).

## Gotchas

- Colab VMs are ephemeral: raw data must be re-downloaded/re-uploaded each session.
- If GPU drops, re-run the failing cell; checkpoints continue from the last saved `best.pt`.
- torch/torchvision versions on Colab are managed by Colab; do not pin them manually.
