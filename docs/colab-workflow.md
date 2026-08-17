# Colab Workflow

All Python runs on Google Colab's cloud runtime. The repo is the source of
truth; the notebooks in `colab/` are thin drivers. Two supported front-ends,
both using the same cloud runtime and the same notebooks:

## Front-end A: VS Code with the Colab extension (recommended)

1. Install the **Colab** extension in VS Code (publisher: Google, id
   `google.colab`), sign in with your Google account.
2. Click the Colab icon → **Connect to a Colab runtime** → choose **T4 GPU**.
3. Open the repo folder in VS Code and open `colab/01_dataset.ipynb` etc.
4. Run cells with ▶ / Shift+Enter.

Notes:

- The cloud computer is ephemeral. Files are uploaded by dragging them INTO
  the Colab extension's file explorer (into `/content`); results are brought
  back by dragging them OUT of the explorer.
- Cell 1 of every notebook is dual-mode: it locates the repo (drag the
  `checkmaize` folder into the explorer, or `git clone`), installs missing
  packages, and prints the GPU status.
- Download cells are dual-mode: in a browser they call `files.download()`; in
  VS Code they print the artifact paths for you to drag out.

## Front-end B: Browser (fallback)

colab.research.google.com → File → Upload notebook → run cells. Uploads go
through the left files pane; `files.download()` sends artifacts to your
browser's Downloads folder.

## Prerequisites per notebook

- Notebook 01: PlantVillage downloads automatically (Hugging Face). The Ghana
  dataset zip (`crop-pest-and-disease-detection.zip` from the Kaggle mirror, or
  `Raw Data.zip` from Mendeley DOI 10.17632/bwh3zbpkpv.1) is uploaded by you;
  the notebook accepts both layouts automatically.
- Notebook 03: set the `winner` variable to the model chosen from the
  Notebook 02 scoreboard.

## Notebooks

1. `colab/01_dataset.ipynb` - extracts PlantVillage + CCMT, builds `raw.csv`,
   splits, tests. Artifact: `splits.zip` -> local `data/manifests/`, commit.
2. `colab/02_train_benchmark.ipynb` - smoke tests, then full 5-model benchmark
   (~2-3 h on T4 GPU). Artifacts: `report.zip` -> `benchmarks/report/` (commit)
   and `runs.zip` -> `artifacts/runs/`. Pick winner.
3. `colab/03_export.ipynb` - export, int8 quantize, verify, app fixtures +
   artifacts. Artifacts: `artifacts.zip` -> `app/assets/model/`,
   `fixtures.zip` -> `app/src/ml/__tests__/fixtures/` (replace), and
   `docs/onnx-contract.md` (already in repo). Commit.

## Gotchas

- The GitHub repo must be **public** — the cloud runtime has no GitHub
  credentials and cannot clone private repos.
- Cloud runtimes are ephemeral: raw data must be re-downloaded/re-uploaded
  each session (the repo itself persists on GitHub / can be re-dragged in).
- If the runtime drops mid-training, reconnect, re-run Cell 1, then re-run the
  training cell — it restarts the interrupted model.
- If Cell 1 prints `GPU available: False`, reconnect choosing a T4 GPU runtime.
- torch/torchvision versions on the runtime are managed by Colab; do not pin
  them manually.
