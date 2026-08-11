# Maize Disease Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research pipeline (dataset → training → RQ3 benchmark → ONNX export → quantization → verification) and an installable Expo SDK 57 mobile app running int8-quantized ONNX inference fully on-device to classify maize leaf diseases (Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy) under Ghanaian field conditions.

**Architecture:** Five one-way-dependent units: `data/` (ingestion + leakage-safe splits) → `training/` (PyTorch fine-tuning on Google Colab) → `benchmarks/` (uniform 5-candidate comparison) → `inference/` (ONNX export, int8 quantization, parity verification) → `app/` (Expo RN/TS, onnxruntime-react-native). The seam between ML and mobile is a pinned ONNX contract (224×224 RGB, ImageNet normalization, class order), enforced by a JS/Python parity test. The user is a JS developer: all Python executes through pre-built Colab notebooks; the user runs cells, downloads artifacts, and relays any error output.

**Tech Stack:** Python 3.10+ (PyTorch 2.x, torchvision, onnx, onnxruntime, scikit-learn, pandas, Pillow, matplotlib, PyYAML, datasets, pytest) on Google Colab free (GPU T4). App: Expo SDK 57 (React Native 0.86, React 19.2.3, TypeScript), onnxruntime-react-native 1.24.3, expo-dev-client, expo-camera, expo-image-picker, expo-image-manipulator, expo-file-system (new `File`/`Paths` API), expo-location, expo-sqlite, expo-sharing, expo-asset, @react-navigation/native + native-stack + bottom-tabs, jest-expo, jpeg-js, pngjs (test-only).

## Global Constraints

- **Class catalog (fixed order):** `["common_rust", "gray_leaf_spot", "northern_leaf_blight", "healthy"]`.
- **ONNX contract (fixed):** input tensor float32 `[1,3,224,224]` CHW; preprocessing = resize to 224×224 (bicubic) then normalize with ImageNet stats mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. Exactly this — enforced by parity tests on both sides.
- **Random seed:** `42` for every random operation (splits, training, benchmark). Determinism is a research requirement.
- **Leakage rule:** no `leaf_id` may appear in both a train manifest and the test manifest of the same experiment.
- **Ghana anchor:** the only Ghanaian-source images are the CCMT dataset (Mendeley DOI `10.17632/bwh3zbpkpv.1`, UENR Sunyani, Ghana), mapped `Leaf blight → northern_leaf_blight`, `Leaf spot → gray_leaf_spot`, `Healthy → healthy`. CCMT contains no Common Rust — the Ghana test set covers 3 classes; this limitation is documented in `docs/dataset-notes.md`.
- **Model candidates (fixed):** `custom_cnn`, `mobilenet_v3_small`, `mobilenet_v3_large`, `efficientnet_b0`, `resnet18` — same data, same seed, same optimizer/schedule.
- **No fp16 anywhere:** Hermes (RN 0.86) has no `Float16Array`; onnxruntime-react-native crashes on fp16 tensors. Models export as fp32; quantization is int8 dynamic.
- **Expo stack (fixed):** Expo SDK 57 (RN 0.86), onnxruntime-react-native pinned `1.24.3` (or latest 1.24.x), Node ≥ 22.13, Android compile/target SDK 36. `onnxruntime-react-native` requires the postinstall fix (delete `unimodule.json`; remove Gradle `VersionNumber` block) — no other autolinking workaround is used, to avoid double package registration.
- **Git hygiene:** raw dataset images and large training artifacts are never committed (`artifacts/`, `data/raw/` gitignored). The one committed exception: `app/assets/model/model_int8.onnx` + `labels.json` + `metrics.json` (required for EAS builds).
- **Python runs on Colab only** — the user has no local Python. Every Python module is committed in the repo; Colab notebooks are thin drivers that sync the repo and run modules.
- **Code style:** no comments in code (repo convention). Tests must be runnable via `pytest` (Python) and `npx jest` (app).
- **Commit message style:** conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), matching the spec commit.

---

### Task 1: Repository scaffold

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `data/`, `data/manifests/`, `data/raw/`, `training/`, `training/configs/`, `benchmarks/`, `benchmarks/report/`, `inference/`, `docs/`, `artifacts/`

**Interfaces:**
- Produces: directory layout consumed by every later task; `requirements.txt` installed by all notebooks.

- [ ] **Step 1: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
node_modules/
.expo/
dist/
android/
ios/
data/raw/
data/manifests/
artifacts/
*.onnx
*.pt
*.zip
.DS_Store
```

- [ ] **Step 2: Write `requirements.txt`**

```
onnx>=1.16.0
onnxruntime>=1.18.0
pandas>=2.0
pyyaml>=6.0
matplotlib>=3.8
scikit-learn>=1.4
pillow>=10.0
datasets>=2.19
tqdm>=4.66
torchinfo>=1.8
pytest>=8.0
```

- [ ] **Step 3: Write root `README.md`**

```markdown
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
```

- [ ] **Step 4: Create directories**

Run: `mkdir -p data/manifests data/raw training/configs benchmarks/report inference docs artifacts`
Expected: directories exist, empty.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md requirements.txt
git commit -m "chore: scaffold repository layout"
```

---

### Task 2: Raw manifest builder (`data/make_manifest.py`)

Builds `data/manifests/raw.csv` from the raw folder layout `data/raw/{source}/{class_folder}/images…`, mapping source-specific folder names to the canonical class names. Produces the single input consumed by the split builder.

**Files:**
- Create: `data/make_manifest.py`
- Test: `data/tests/test_make_manifest.py`

**Interfaces:**
- Produces: `build_raw_manifest(raw_root: Path, out_csv: Path) -> list[dict]` with rows `{path, source, leaf_id, class}`; `path` relative to `raw_root`.

- [ ] **Step 1: Write the failing test**

```python
import csv
from pathlib import Path
import tempfile

from data.make_manifest import build_raw_manifest

def _make_tree(root: Path) -> None:
    pv = root / "plantvillage" / "Common_rust_"
    pv.mkdir(parents=True)
    (pv / "leafA__1.jpg").touch()
    (pv / "leafA__2.jpg").touch()
    g = root / "ccmt_ghana" / "Leaf blight"
    g.mkdir(parents=True)
    (g / "blight_01.jpg").touch()

def test_build_raw_manifest_maps_and_groups():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root)
        out = root / "raw.csv"
        rows = build_raw_manifest(root, out)
        assert len(rows) == 3
        by_class = {r["class"] for r in rows}
        assert by_class == {"common_rust", "northern_leaf_blight"}
        cr = [r for r in rows if r["class"] == "common_rust"]
        assert {r["leaf_id"] for r in cr} == {"leafA"}
        assert all(r["path"].endswith(".jpg") for r in rows)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert [r["class"] for r in reader] == [r["class"] for r in rows]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest data/tests/test_make_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.make_manifest'`.

- [ ] **Step 3: Write the implementation**

```python
import argparse
import csv
from pathlib import Path

SOURCE_FOLDER_MAP = {
    "plantvillage": {
        "Cercospora_leaf_spot Gray_leaf_spot": "gray_leaf_spot",
        "Common_rust_": "common_rust",
        "Northern_Leaf_Blight": "northern_leaf_blight",
        "healthy": "healthy",
    },
    "ccmt_ghana": {
        "Leaf blight": "northern_leaf_blight",
        "Leaf spot": "gray_leaf_spot",
        "Healthy": "healthy",
    },
}


def build_raw_manifest(raw_root: Path, out_csv: Path) -> list[dict]:
    rows: list[dict] = []
    for source, folder_map in SOURCE_FOLDER_MAP.items():
        source_dir = raw_root / source
        if not source_dir.exists():
            continue
        for folder, class_name in folder_map.items():
            class_dir = source_dir / folder
            if not class_dir.exists():
                continue
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                leaf_id = img.name.split("__")[0] if source == "plantvillage" else img.stem
                rows.append(
                    {
                        "path": str(img.relative_to(raw_root)),
                        "source": source,
                        "leaf_id": leaf_id,
                        "class": class_name,
                    }
                )
    if not rows:
        raise ValueError(f"no images found under {raw_root}; expected data/raw/<source>/<class_folder>")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/raw.csv"))
    args = parser.parse_args()
    rows = build_raw_manifest(args.raw_root, args.out)
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"wrote {len(rows)} rows to {args.out}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest data/tests/test_make_manifest.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add data/make_manifest.py data/tests/test_make_manifest.py
git commit -m "feat: add raw manifest builder with source class mapping"
```

---

### Task 3: Leakage-safe split builder (`data/make_splits.py`)

Builds the five manifests from `raw.csv`, implementing the approved two-axis evaluation: within-source (Ghana holdout test) and domain-shift (train on PlantVillage only, test on all Ghana images). Splits respect `leaf_id` groups and are deterministic under seed 42.

**Files:**
- Create: `data/make_splits.py`
- Test: `data/tests/test_make_splits.py`

**Interfaces:**
- Consumes: `data/manifests/raw.csv` rows `{path, source, leaf_id, class}` (Task 2).
- Produces, under `data/manifests/`:
  - `train.csv`, `val.csv`, `test.csv` (main experiment; `test.csv` = Ghana holdout only)
  - `domain_shift_train.csv`, `domain_shift_val.csv`, `domain_shift_test.csv` (all Ghana images)
  - Rows keep the same columns as `raw.csv`.

- [ ] **Step 1: Write the failing test**

```python
import csv
from pathlib import Path
import tempfile

from data.make_splits import build_splits

def _raw_rows(tmp: Path) -> None:
    rows = []
    for i in range(40):
        rows.append({"path": f"ccmt_ghana/Leaf blight/{i}.jpg", "source": "ccmt_ghana", "leaf_id": f"g{i}", "class": "northern_leaf_blight"})
    for i in range(40):
        rows.append({"path": f"ccmt_ghana/Healthy/{i}.jpg", "source": "ccmt_ghana", "leaf_id": f"h{i}", "class": "healthy"})
    for i in range(40):
        rows.append({"path": f"ccmt_ghana/Leaf spot/{i}.jpg", "source": "ccmt_ghana", "leaf_id": f"s{i}", "class": "gray_leaf_spot"})
    for i in range(120):
        rows.append({"path": f"plantvillage/Common_rust_/{i}.jpg", "source": "plantvillage", "leaf_id": f"c{i}", "class": "common_rust"})
    with (tmp / "raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        writer.writeheader()
        writer.writerows(rows)

def _read(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))

def test_splits_respect_leakage_and_axes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _raw_rows(tmp)
        build_splits(tmp / "raw.csv", tmp, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
        train = _read(tmp / "train.csv")
        val = _read(tmp / "val.csv")
        test = _read(tmp / "test.csv")
        d_train = _read(tmp / "domain_shift_train.csv")
        d_test = _read(tmp / "domain_shift_test.csv")
        assert test, "test.csv must not be empty"
        assert all(r["source"] == "ccmt_ghana" for r in test), "main test must be Ghana-only"
        train_ids = {(r["source"], r["leaf_id"]) for r in train}
        test_ids = {(r["source"], r["leaf_id"]) for r in test}
        assert not (train_ids & test_ids), "leaf leakage between train and test"
        assert all(r["source"] == "plantvillage" for r in d_train), "domain-shift train is PlantVillage-only"
        ghana_total = sum(1 for r in train + val + test if r["source"] == "ccmt_ghana")
        assert ghana_total == 120
        assert len(d_test) == 120, "domain-shift test = all Ghana images"
        classes_in_test = {r["class"] for r in test}
        assert classes_in_test == {"northern_leaf_blight", "healthy", "gray_leaf_spot"}

def test_splits_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _raw_rows(tmp)
        build_splits(tmp / "raw.csv", tmp, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
        first = (tmp / "test.csv").read_bytes()
        build_splits(tmp / "raw.csv", tmp, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
        assert (tmp / "test.csv").read_bytes() == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest data/tests/test_make_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.make_splits'`.

- [ ] **Step 3: Write the implementation**

```python
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def _read(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _write(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        writer.writeheader()
        writer.writerows(rows)


def _group_by_leaf(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["leaf_id"])].append(r)
    return groups


def _split_groups(groups: list[list[dict]], rng: random.Random, ratios: list[float]) -> list[list[list[dict]]]:
    per_class: dict[str, list[list[dict]]] = defaultdict(list)
    for g in groups:
        per_class[g[0]["class"]].append(g)
    assigned: list[list[list[dict]]] = [[] for _ in ratios]
    for class_groups in per_class.values():
        rng.shuffle(class_groups)
        total = len(class_groups)
        start = 0
        for idx, ratio in enumerate(ratios):
            end = start + int(round(total * ratio)) if idx < len(ratios) - 1 else total
            assigned[idx].extend(class_groups[start:end])
            start = end
    return assigned


def build_splits(raw_csv: Path, out_dir: Path, seed: int, ghana_test_ratio: float, val_ratio: float) -> None:
    rows = _read(raw_csv)
    rng = random.Random(seed)
    ghana = [r for r in rows if r["source"] == "ccmt_ghana"]
    pv = [r for r in rows if r["source"] == "plantvillage"]
    if not ghana:
        raise ValueError("no ccmt_ghana rows found; Ghana test set cannot be built")
    g_test, g_val, g_train = _split_groups(
        list(_group_by_leaf(ghana).values()), rng, [ghana_test_ratio, val_ratio, 1.0 - ghana_test_ratio - val_ratio]
    )
    pv_val, pv_train = _split_groups(list(_group_by_leaf(pv).values()), rng, [val_ratio, 1.0 - val_ratio])
    flat = lambda groups: [r for g in groups for r in g]
    _write(out_dir / "train.csv", flat(g_train) + flat(pv_train))
    _write(out_dir / "val.csv", flat(g_val) + flat(pv_val))
    _write(out_dir / "test.csv", flat(g_test))
    _write(out_dir / "domain_shift_train.csv", flat(pv_train) + flat(pv_val))
    _write(out_dir / "domain_shift_val.csv", flat(pv_val))
    _write(out_dir / "domain_shift_test.csv", flat(g_train) + flat(g_val) + flat(g_test))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/manifests/raw.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ghana-test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    build_splits(args.raw, args.out, args.seed, args.ghana_test_ratio, args.val_ratio)
    for name in ["train", "val", "test", "domain_shift_train", "domain_shift_val", "domain_shift_test"]:
        print(f"{name}: {len(_read(args.out / (name + '.csv')))} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest data/tests/test_make_splits.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data/make_splits.py data/tests/test_make_splits.py
git commit -m "feat: add leakage-safe split builder with domain-shift axis"
```

---

### Task 4: Dataset acquisition notebook (Colab)

A thin Colab driver the user runs: syncs the repo, downloads PlantVillage (via Hugging Face, no account), takes the user's browser-uploaded CCMT zip, builds `data/raw`, runs Tasks 2–3, runs their tests, and downloads the split manifests.

**Files:**
- Create: `colab/01_dataset.ipynb` (author as a Markdown cell script — notebook JSON is generated below)

**Interfaces:**
- Consumes: repo sync (git clone or zip upload); CCMT `Raw Data.zip` uploaded by the user via the Colab file browser.
- Produces: `/content/data/raw/...` images (ephemeral), `data/manifests/*.csv`, and a downloaded `splits.zip`.

- [ ] **Step 1: Create the notebook cells**

Create a new Colab notebook and paste these cells verbatim. Cell 1 — repo sync:

```python
import os
if os.path.exists('/content/checkmaize'):
    os.system('cd /content/checkmaize && git pull')
else:
    print("OPTION A: push this repo to GitHub and run: !git clone <your-repo-url> /content/checkmaize")
    print("OPTION B (no GitHub): zip the local repo (excluding node_modules, .git, artifacts) and upload to /content/checkmaize.zip, then:")
    print("  !mkdir -p /content/checkmaize && !unzip -q /content/checkmaize.zip -d /content/checkmaize")
print("Then run: !pip install -q -r /content/checkmaize/requirements.txt")
```

Cell 2 — extract PlantVillage maize (Hugging Face):

```python
import os
os.chdir('/content/checkmaize')
from datasets import load_dataset
from PIL import Image
import shutil

dst = 'data/raw/plantvillage'
shutil.rmtree(dst, ignore_errors=True)
CLASS_MAP = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_': 'Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern_Leaf_Blight',
    'Corn_(maize)___healthy': 'healthy',
}
os.makedirs(dst, exist_ok=True)
for split in ['train', 'test']:
    ds = load_dataset('mohanty/PlantVillage', split=split)
    corn = [r for r in ds if r['crop'] == 'Corn (maize)']
    print(f"{split}: {len(corn)} corn rows")
    for r in corn:
        folder = CLASS_MAP[r['label']]
        out = os.path.join(dst, folder, f"{r['leaf_id']}__{r['image_path'].rsplit('/', 1)[-1]}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        r['image'].save(out)
print("plantvillage extraction done")
```

Cell 3 — unpack CCMT upload:

```python
import os, zipfile, shutil
os.chdir('/content/checkmaize')
if not os.path.exists('/content/data/raw'):
    os.makedirs('/content/data/raw', exist_ok=True)
if os.path.exists('/content/Raw Data.zip'):
    with zipfile.ZipFile('/content/Raw Data.zip') as z:
        z.extractall('/content/ccmt_extract')
    src = None
    for root, dirs, files in os.walk('/content/ccmt_extract'):
        if 'Maize' in dirs:
            src = os.path.join(root, 'Maize')
            break
    assert src, "Maize folder not found in upload"
    dst = 'data/raw/ccmt_ghana'
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    for name in sorted(os.listdir(dst)):
        print(name, len(os.listdir(os.path.join(dst, name))))
else:
    print("No /content/Raw Data.zip found. Upload it from the Mendeley page "
          "(DOI 10.17632/bwh3zbpkpv.1, download 'Raw Data.zip') using the files pane, then re-run this cell.")
```

Cell 4 — build manifest + splits + tests:

```python
os.chdir('/content/checkmaize')
!python -m data.make_manifest
!python -m data.make_splits
!python -m pytest data/tests -v
```

Cell 5 — download artifacts:

```python
import shutil, os
shutil.make_archive('/content/splits', 'zip', 'data/manifests')
from google.colab import files
files.download('/content/splits.zip')
```

- [ ] **Step 2: Author the notebook file for the repo**

Run locally: `jupyter nbconvert` is not required. Create `colab/01_dataset.ipynb` containing the 5 cells above (Cell 1 as code with printed instructions, Cells 2–5 as code cells). Commit the notebook.

- [ ] **Step 3: Commit**

```bash
git add colab/01_dataset.ipynb
git commit -m "feat: add dataset acquisition colab notebook"
```

**Execution gate (user):** After running Cell 4, expected output includes `wrote N rows to data/manifests/raw.csv` with per-class counts and `2 passed` from pytest. The 3-class Ghana counts come from CCMT (`northern_leaf_blight` ≈ 1,006, `gray_leaf_spot` ≈ 1,259, `healthy` ≈ 208); PlantVillage contributes ~4,000 across 4 classes. Save `splits.zip` into the local `data/manifests/` folder and commit it.

- [ ] **Step 4: Commit manifests**

```bash
git add data/manifests
git commit -m "data: add split manifests (seed 42)"
```

---

### Task 5: Dataset loaders and transforms (`training/data.py`)

Loaders that read manifest CSVs, plus the transform contract: eval transforms are exactly the ONNX contract (resize 224×224, ImageNet normalize); train transforms add field-condition augmentation. Shared by all training and verification code.

**Files:**
- Create: `training/data.py`
- Test: `training/tests/test_data.py`

**Interfaces:**
- Consumes: manifest CSVs from Task 3.
- Produces:
  - `MEAN: tuple[float, float, float]`, `STD: tuple[float, float, float]`
  - `train_transforms() -> torchvision.transforms.Compose`
  - `eval_transforms() -> torchvision.transforms.Compose`
  - `ManifestDataset(csv_path: Path, root: Path, transform=None, limit: int | None = None)` — `__len__`, `__getitem__ -> (tensor, int)` where `int` is the class index in `CLASSES` order.
  - `CLASSES: list[str]`

- [ ] **Step 1: Write the failing test**

```python
import csv
from pathlib import Path
import tempfile
import numpy as np
import torch
from PIL import Image

from training.data import ManifestDataset, CLASSES, eval_transforms

def _make_csv(tmp: Path, n: int = 8) -> Path:
    raw = tmp / "img"
    raw.mkdir()
    rows = []
    for i in range(n):
        cls = CLASSES[i % len(CLASSES)]
        folder = raw / cls
        folder.mkdir(exist_ok=True)
        img = Image.fromarray(np.full((240, 320, 3), 120 + i, dtype=np.uint8))
        p = folder / f"{i}.jpg"
        img.save(p)
        rows.append({"path": f"img/{cls}/{i}.jpg", "source": "ccmt_ghana", "leaf_id": str(i), "class": cls})
    csv_path = tmp / "rows.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "leaf_id", "class"])
        w.writeheader()
        w.writerows(rows)
    return csv_path

def test_dataset_and_contract():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv_path = _make_csv(tmp)
        ds = ManifestDataset(csv_path, root=tmp)
        assert len(ds) == 8
        x, y = ds[0]
        assert tuple(x.shape) == (3, 224, 224)
        assert x.dtype == torch.float32
        assert 0 <= y < len(CLASSES)

def test_eval_transforms_match_contract():
    img = Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8))
    x = eval_transforms()(img)
    assert tuple(x.shape) == (3, 224, 224)
    assert torch.allclose(x, torch.tensor((128 / 255 - 0.485) / 0.229).expand_as(x), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest training/tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.data'`.

- [ ] **Step 3: Write the implementation**

```python
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["common_rust", "gray_leaf_spot", "northern_leaf_blight", "healthy"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def eval_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, transform=None, limit: int | None = None):
        self.root = Path(root)
        self.transform = transform if transform is not None else eval_transforms()
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        if limit:
            rows = rows[:limit]
        self.paths = [self.root / r["path"] for r in rows]
        self.labels = [CLASSES.index(r["class"]) for r in rows]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest training/tests/test_data.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add training/data.py training/tests/test_data.py
git commit -m "feat: add manifest dataset and contract transforms"
```

---

### Task 6: Training driver (`training/train.py`) + configs

Generic transfer-learning fine-tune driver driven by a YAML config: builds the model (torchvision pretrained backbones or the custom CNN), trains with AdamW + cosine schedule, saves `best.pt` and `metrics.json` (val accuracy, macro-F1, per-class precision/recall, confusion matrix via scikit-learn). Deterministic under seed 42.

**Files:**
- Create: `training/train.py`
- Create: `training/configs/custom_cnn.yaml`
- Create: `training/configs/mobilenet_v3_small.yaml`
- Create: `training/configs/mobilenet_v3_large.yaml`
- Create: `training/configs/efficientnet_b0.yaml`
- Create: `training/configs/resnet18.yaml`
- Create: `training/tests/test_train.py`

**Interfaces:**
- Consumes: `training/data.py` (Task 5); manifests (Task 3).
- Produces: `build_model(arch: str, num_classes: int) -> nn.Module`; `main(argv)` reading `--config` (YAML) and `--out-dir`; writes `{out_dir}/best.pt` (`{"state_dict", "arch", "class_names", "config"}`) and `{out_dir}/metrics.json`.

**Config schema:** `arch`, `epochs`, `batch_size`, `lr`, `weight_decay`, `train_manifest`, `val_manifest`, `test_manifest`, `data_root`, `seed`.

- [ ] **Step 1: Write the failing test**

```python
import json
import tempfile
from pathlib import Path
import yaml

from training.train import main

def test_train_smoke(tmp_path: Path):
    cfg = {
        "arch": "custom_cnn",
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "data_root": str(tmp_path),
        "seed": 42,
        "train_manifest": "train.csv",
        "val_manifest": "val.csv",
        "test_manifest": "test.csv",
    }
    from data.make_manifest import build_raw_manifest
    from data.make_splits import build_splits
    import csv
    import numpy as np
    from PIL import Image
    raw = tmp_path / "raw"
    (raw / "plantvillage" / "Common_rust_").mkdir(parents=True)
    (raw / "ccmt_ghana" / "Leaf blight").mkdir(parents=True)
    (raw / "ccmt_ghana" / "Leaf spot").mkdir(parents=True)
    (raw / "ccmt_ghana" / "Healthy").mkdir(parents=True)
    for i in range(30):
        img = Image.fromarray(np.full((224, 224, 3), 100 + i, dtype=np.uint8))
        img.save(raw / "plantvillage" / "Common_rust_" / f"c{i}.jpg")
        img.save(raw / "ccmt_ghana" / "Leaf blight" / f"g{i}.jpg")
        img.save(raw / "ccmt_ghana" / "Leaf spot" / f"g2_{i}.jpg")
        img.save(raw / "ccmt_ghana" / "Healthy" / f"g3_{i}.jpg")
    build_raw_manifest(raw, tmp_path / "raw.csv")
    build_splits(tmp_path / "raw.csv", tmp_path, seed=42, ghana_test_ratio=0.2, val_ratio=0.1)
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w") as f:
        yaml.dump(cfg, f)
    out = tmp_path / "out"
    main(["--config", str(cfg_path), "--out-dir", str(out)])
    assert (out / "best.pt").exists()
    with (out / "metrics.json").open() as f:
        metrics = json.load(f)
    assert "accuracy" in metrics and "macro_f1" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest training/tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train'`.

- [ ] **Step 3: Write the implementation**

```python
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from training.data import CLASSES, ManifestDataset, eval_transforms, train_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(64 * 56 * 56, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x.flatten(1))


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "custom_cnn":
        return SimpleCNN(num_classes)
    weights = torchvision.models.get_model_weights(arch)
    model = torchvision.models.get_model(arch, weights=weights)
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASSES))), average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class": {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for cls, p, r, f in zip(CLASSES, precision, recall, f1)
        },
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES)))).tolist(),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["data_root"])
    train_ds = ManifestDataset(root / cfg["train_manifest"], root, transform=train_transforms())
    val_ds = ManifestDataset(root / cfg["val_manifest"], root)
    test_ds = ManifestDataset(root / cfg["test_manifest"], root)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    model = build_model(cfg["arch"], len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{cfg['epochs']} val_acc={val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(
                {"state_dict": model.state_dict(), "arch": cfg["arch"], "class_names": CLASSES, "config": cfg},
                args.out_dir / "best.pt",
            )
    model.load_state_dict(torch.load(args.out_dir / "best.pt")["state_dict"])
    test_metrics = _evaluate(model, test_loader, device)
    test_metrics["best_val_accuracy"] = best_acc
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"test_acc={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the five configs**

`training/configs/custom_cnn.yaml`:

```yaml
arch: custom_cnn
epochs: 15
batch_size: 32
lr: 0.003
weight_decay: 0.0
data_root: data
train_manifest: train.csv
val_manifest: val.csv
test_manifest: test.csv
seed: 42
```

`training/configs/mobilenet_v3_small.yaml`:

```yaml
arch: mobilenet_v3_small
epochs: 8
batch_size: 32
lr: 0.001
weight_decay: 0.0001
data_root: data
train_manifest: train.csv
val_manifest: val.csv
test_manifest: test.csv
seed: 42
```

`training/configs/mobilenet_v3_large.yaml`:

```yaml
arch: mobilenet_v3_large
epochs: 8
batch_size: 32
lr: 0.001
weight_decay: 0.0001
data_root: data
train_manifest: train.csv
val_manifest: val.csv
test_manifest: test.csv
seed: 42
```

`training/configs/efficientnet_b0.yaml`:

```yaml
arch: efficientnet_b0
epochs: 8
batch_size: 32
lr: 0.001
weight_decay: 0.0001
data_root: data
train_manifest: train.csv
val_manifest: val.csv
test_manifest: test.csv
seed: 42
```

`training/configs/resnet18.yaml`:

```yaml
arch: resnet18
epochs: 8
batch_size: 32
lr: 0.001
weight_decay: 0.0001
data_root: data
train_manifest: train.csv
val_manifest: val.csv
test_manifest: test.csv
seed: 42
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /content && python -m pytest training/tests/test_train.py -v`
Expected: PASS (1 passed). The smoke run trains 1 epoch on ~120 tiny images on CPU.

- [ ] **Step 6: Commit**

```bash
git add training/train.py training/configs training/tests/test_train.py
git commit -m "feat: add deterministic fine-tuning driver and model configs"
```

---

### Task 7: RQ3 benchmark driver (`benchmarks/compare.py`)

Runs the five candidate configs through `training/train.py`, collects `metrics.json` from each, appends parameter counts and model size (fp32 ONNX bytes after export via `inference/export.py` from Task 9), and emits the comparison table + accuracy plot into `benchmarks/report/`. Runs on the Colab GPU.

**Files:**
- Create: `benchmarks/compare.py`
- Test: `benchmarks/tests/test_compare.py`

**Interfaces:**
- Consumes: `training/train.py` (Task 6), `inference/export.py::export_onnx` (Task 9), configs (Task 6).
- Produces: `benchmarks/report/comparison.csv` (rows: `model,accuracy,macro_f1,params,onnx_bytes`), `benchmarks/report/accuracy.png`, `benchmarks/report/comparison.md`, prints a Markdown table to stdout.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest benchmarks/tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.compare'`.

- [ ] **Step 3: Write the implementation**

```python
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from training.train import build_model
from training.data import CLASSES

MODELS = ["custom_cnn", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]


def count_params(arch: str) -> int:
    model = build_model(arch, len(CLASSES))
    return sum(p.numel() for p in model.parameters())


def run_training(config_path: Path, out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "training.train", "--config", str(config_path), "--out-dir", str(out_dir)],
        check=True,
    )


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
    for m in MODELS:
        p = args.runs_dir / m / "model.onnx"
        onnx_bytes[m] = p.stat().st_size if p.exists() else 0
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest benchmarks/tests/test_compare.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare.py benchmarks/tests/test_compare.py
git commit -m "feat: add uniform RQ3 benchmark driver"
```

---

### Task 8: Training + benchmark notebook (Colab)

Drives Tasks 6–7 on the Colab GPU: runs the smoke tests, then the full 5-candidate benchmark (custom CNN first for a fast fail), downloads the report and all run artifacts.

**Files:**
- Create: `colab/02_train_benchmark.ipynb`

**Interfaces:**
- Consumes: repo sync; `data/manifests/*.csv` (committed in Task 4); configs (Task 6); `inference/export.py` (Task 9 — imported by compare.py).
- Produces: `benchmarks/report/*` committed back; `artifacts/runs/*` downloaded to local `artifacts/runs/` (gitignored).

- [ ] **Step 1: Create the notebook cells**

Cell 1 — repo sync + installs (same as Task 4 Cell 1, plus): `!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121` if torch is missing on the GPU runtime; then `!python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"` — expected `True`.

Cell 2 — smoke tests:

```python
os.chdir('/content/checkmaize')
!python -m pytest training/tests benchmarks/tests -v
```

Cell 3 — full benchmark:

```python
os.chdir('/content/checkmaize')
!python -m benchmarks.compare
```

Expected: five `=== training <model> ===` blocks, then the Markdown comparison table. Estimated wall time on a T4: custom_cnn ~10 min, the four transfer-learned models ~25–40 min each.

Cell 4 — download artifacts:

```python
import os, shutil
os.chdir('/content/checkmaize')
shutil.make_archive('/content/report', 'zip', 'benchmarks/report')
shutil.make_archive('/content/runs', 'zip', 'artifacts/runs')
from google.colab import files
files.download('/content/report.zip')
files.download('/content/runs.zip')
```

- [ ] **Step 2: Commit the notebook**

```bash
git add colab/02_train_benchmark.ipynb
git commit -m "feat: add training and benchmark colab notebook"
```

**Execution gate (user + AI):** Unzip `report.zip` into local `benchmarks/report/` and `runs.zip` into local `artifacts/runs/`. The AI explains the table (accuracy vs size) and recommends the winner (expected: `efficientnet_b0` or `mobilenet_v3_small`). The user confirms the pick; commit the report:

- [ ] **Step 3: Commit report**

```bash
git add benchmarks/report
git commit -m "data: add RQ3 comparison report"
```

---

### Task 9: ONNX export (`inference/export.py`)

Freezes a trained checkpoint to a fixed-batch fp32 ONNX file (opset 17, input `[1,3,224,224]`, dynamic batch axis) and generates `docs/onnx-contract.md` documenting the exact contract.

**Files:**
- Create: `inference/export.py`
- Test: `inference/tests/test_export.py`

**Interfaces:**
- Consumes: `training/train.py::build_model` (Task 6); `best.pt` checkpoint.
- Produces:
  - `export_onnx(checkpoint: Path, out_path: Path, opset: int = 17) -> None`
  - `write_contract(out_path: Path, label_names: list[str]) -> None` (writes `docs/onnx-contract.md`)

- [ ] **Step 1: Write the failing test**

```python
import torch
from pathlib import Path

from inference.export import export_onnx, write_contract

def test_export_produces_runnable_session(tmp_path: Path):
    import json
    from training.train import build_model
    from training.data import CLASSES
    model = build_model("custom_cnn", len(CLASSES))
    ckpt = {"state_dict": model.state_dict(), "arch": "custom_cnn", "class_names": CLASSES, "config": {}}
    torch.save(ckpt, tmp_path / "best.pt")
    out = tmp_path / "model.onnx"
    export_onnx(tmp_path / "best.pt", out)
    assert out.exists() and out.stat().st_size > 0
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = {"input": torch.rand(1, 3, 224, 224).numpy()}
    y = sess.run(None, x)
    assert y[0].shape == (1, 4)

def test_contract_doc_contains_stats(tmp_path: Path):
    from training.data import CLASSES
    doc = tmp_path / "contract.md"
    write_contract(doc, CLASSES)
    text = doc.read_text()
    assert "0.485" in text and "224" in text and "common_rust" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest inference/tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.export'`.

- [ ] **Step 3: Write the implementation**

```python
import argparse
from pathlib import Path

import torch

from training.data import CLASSES, MEAN, STD
from training.train import build_model


def export_onnx(checkpoint: Path, out_path: Path, opset: int = 17) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


def write_contract(out_path: Path, label_names: list[str]) -> None:
    mean = ", ".join(str(x) for x in MEAN)
    std = ", ".join(str(x) for x in STD)
    out_path.write_text(
        f"""# ONNX Model Contract

This file is generated by `inference/export.py`. The app MUST match these
values exactly; the parity test in `app/src/ml/__tests__/preprocess.test.ts`
enforces it.

## Input

- Tensor name: `input`
- Shape: `[1, 3, 224, 224]` (batch 1, CHW)
- Dtype: float32
- Values: RGB pixels resized to 224x224 (bicubic), then normalized:
  `(pixel / 255 - mean) / std` per channel.

## Normalization stats

- mean: [{mean}]
- std: [{std}]

## Output

- Tensor name: `output`
- Shape: `[1, {len(label_names)}]`, logits (pre-softmax)
- Class index order: {label_names}

## Runtime

- ONNX opset 17, fp32, int8 dynamic quantization for the shipped model.
- No float16 tensors anywhere (Hermes/RN 0.86 has no Float16Array).
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("docs/onnx-contract.md"))
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.out)
    write_contract(args.contract, CLASSES)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest inference/tests/test_export.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/export.py inference/tests/test_export.py
git commit -m "feat: add onnx export and contract generation"
```

---

### Task 10: int8 quantization (`inference/quantize.py`)

Applies onnxruntime dynamic int8 quantization (weights → QUInt8, activations stay fp32) to the fp32 export.

**Files:**
- Create: `inference/quantize.py`
- Test: `inference/tests/test_quantize.py`

**Interfaces:**
- Consumes: `model.onnx` (Task 9).
- Produces: `quantize(fp32_path: Path, out_path: Path) -> None` (writes `model_int8.onnx`).

- [ ] **Step 1: Write the failing test**

```python
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
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    import numpy as np
    y = sess.run(None, {"input": np.ones((1, 3), dtype=np.float32)})
    assert y[0].shape == (1, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest inference/tests/test_quantize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.quantize'`.

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest inference/tests/test_quantize.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/quantize.py inference/tests/test_quantize.py
git commit -m "feat: add int8 dynamic quantization"
```

---

### Task 11: Verification (`inference/verify.py`)

Three gates plus the app parity fixture: (1) PyTorch ↔ ONNX fp32 parity on a sample of the Ghana test set (max logit diff < 1e-3), (2) int8 vs fp32 accuracy on the full Ghana test set (shipped only if drop ≤ 0.01), (3) generates the deterministic 32×32 PNG + reference tensor JSON the app's jest parity test consumes.

**Files:**
- Create: `inference/verify.py`
- Test: `inference/tests/test_verify.py`

**Interfaces:**
- Consumes: `best.pt`, `model.onnx`, `model_int8.onnx` (Tasks 9–10), `data/manifests/test.csv`.
- Produces:
  - `verify(checkpoint, fp32_path, int8_path, test_manifest, data_root, fixture_out_dir) -> dict` (report with `parity_max_diff`, `fp32_accuracy`, `int8_accuracy`, `delta`, `ship_int8: bool`)
  - Writes `inference/verify_report.json` and fixture files `app/src/ml/__tests__/fixtures/sample.png` + `reference_tensor.json` (32×32, CHW-normalized floats, 3072 values).

- [ ] **Step 1: Write the failing test**

```python
import json
import numpy as np
import torch
from pathlib import Path

from inference.verify import generate_fixture

def test_fixture_roundtrip(tmp_path: Path):
    png, tensor_path = generate_fixture(tmp_path)
    assert png.exists() and tensor_path.exists()
    ref = json.loads(tensor_path.read_text())
    assert len(ref) == 32 * 32 * 3
    from PIL import Image
    img = np.array(Image.open(png).convert("RGB")).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)).flatten()
    img = (img - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
    assert np.allclose(img.flatten(), np.array(ref), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /content && python -m pytest inference/tests/test_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.verify'`.

- [ ] **Step 3: Write the implementation**

```python
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from training.data import CLASSES, ManifestDataset, MEAN, STD
from training.train import build_model


def _top1(sess, loader) -> float:
    preds, labels = [], []
    for x, y in loader:
        logits = sess.run(None, {"input": x.numpy()})[0]
        preds.extend(logits.argmax(axis=1).tolist())
        labels.extend(y.tolist())
    return accuracy_score(labels, preds)


def verify(checkpoint, fp32_path, int8_path, test_manifest, data_root) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_ds = ManifestDataset(test_manifest, Path(data_root), limit=64)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    torch_logits, onnx_logits = [], []
    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for x, _ in loader:
            torch_logits.append(model(x).numpy())
            onnx_logits.append(fp32_sess.run(None, {"input": x.numpy()})[0])
    torch_logits = np.concatenate(torch_logits)
    onnx_logits = np.concatenate(onnx_logits)
    parity_max_diff = float(np.abs(torch_logits - onnx_logits).max())
    full_ds = ManifestDataset(test_manifest, Path(data_root))
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)
    fp32_acc = _top1(fp32_sess, full_loader)
    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8_acc = _top1(int8_sess, full_loader)
    delta = int8_acc - fp32_acc
    report = {
        "parity_max_diff": parity_max_diff,
        "parity_pass": parity_max_diff < 1e-3,
        "fp32_accuracy": fp32_acc,
        "int8_accuracy": int8_acc,
        "delta": delta,
        "ship_int8": delta >= -0.01,
    }
    return report


def generate_fixture(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0, 255, 32, dtype=np.uint8)
    x = np.linspace(255, 0, 32, dtype=np.uint8)
    r = np.broadcast_to(y[None, :], (32, 32))
    g = np.broadcast_to(x[:, None], (32, 32))
    b = (r.astype(np.uint16) + g.astype(np.uint16)) // 2
    img = np.stack([r, g, b.astype(np.uint8)], axis=-1)
    png = out_dir / "sample.png"
    Image.fromarray(img).save(png)
    rgb = img.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    norm = (chw - np.array(MEAN).reshape(3, 1, 1)) / np.array(STD).reshape(3, 1, 1)
    tensor_path = out_dir / "reference_tensor.json"
    tensor_path.write_text(json.dumps(norm.flatten().tolist()))
    return png, tensor_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--report-out", type=Path, default=Path("inference/verify_report.json"))
    parser.add_argument("--fixture-out", type=Path, default=Path("app/src/ml/__tests__/fixtures"))
    args = parser.parse_args()
    report = verify(args.checkpoint, args.fp32, args.int8, args.test_manifest, args.data_root)
    args.report_out.write_text(json.dumps(report, indent=2))
    generate_fixture(args.fixture_out)
    print(json.dumps(report, indent=2))
    if not report["parity_pass"]:
        raise SystemExit("FAIL: fp32 export parity exceeds tolerance")
    if not report["ship_int8"]:
        print("WARNING: int8 accuracy drop > 1%; ship fp32 instead")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /content && python -m pytest inference/tests/test_verify.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/verify.py inference/tests/test_verify.py
git commit -m "feat: add parity, quantization delta and fixture generation"
```

---

### Task 12: Export + verify notebook (Colab)

Runs export → quantize → verify for the benchmark winner and produces the app artifacts: `model_int8.onnx` (or fp32 fallback per the gate), `labels.json`, `metrics.json`, the contract doc, and the jest parity fixtures.

**Files:**
- Create: `colab/03_export.ipynb`

**Interfaces:**
- Consumes: `artifacts/runs/{winner}/best.pt` + `model.onnx` (Task 8), `benchmarks/report/comparison.csv` (Task 8).
- Produces: `artifacts/model_int8.onnx`, `artifacts/labels.json`, `artifacts/metrics.json`, `docs/onnx-contract.md`, `app/src/ml/__tests__/fixtures/{sample.png,reference_tensor.json}` — downloaded by the user into the local repo and committed in Task 20.

- [ ] **Step 1: Create the notebook cells**

Cell 1 — repo sync + installs (same as Task 4 Cell 1, plus `!pip install -q onnx onnxruntime` if not present).

Cell 2 — set the winner:

```python
import os
os.chdir('/content/checkmaize')
winner = 'efficientnet_b0'  # SET THIS to the model chosen from the Task 8 comparison table
assert os.path.exists(f'artifacts/runs/{winner}/best.pt'), f'no checkpoint for {winner}'
print('winner:', winner)
```

Cell 3 — export, quantize, verify:

```python
os.chdir('/content/checkmaize')
!python -m inference.export --checkpoint artifacts/runs/{winner}/best.pt --out artifacts/runs/{winner}/model.onnx
!python -m inference.quantize --fp32 artifacts/runs/{winner}/model.onnx --out artifacts/model_int8.onnx
!python -m inference.verify --checkpoint artifacts/runs/{winner}/best.pt --fp32 artifacts/runs/{winner}/model.onnx --int8 artifacts/model_int8.onnx
```

Expected: verify report printed with `parity_max_diff` < 0.001, `delta` ≥ −0.01, `ship_int8: true`. If `ship_int8` is false, ship the fp32 export instead (rename `artifacts/runs/{winner}/model.onnx` → `artifacts/model.onnx` and use that path everywhere downstream).

Cell 4 — app artifacts (`labels.json`, `metrics.json`):

```python
import csv, json, os
os.chdir('/content/checkmaize')
labels = ["common_rust", "gray_leaf_spot", "northern_leaf_blight", "healthy"]
with open('artifacts/labels.json', 'w') as f:
    json.dump(labels, f)
with open('artifacts/runs/' + winner + '/metrics.json') as f:
    m = json.load(f)
with open('benchmarks/report/comparison.csv') as f:
    row = [r for r in csv.DictReader(f) if r['model'] == winner][0]
with open('inference/verify_report.json') as f:
    v = json.load(f)
metrics = {
    "model": winner,
    "test_accuracy": m["accuracy"],
    "macro_f1": m["macro_f1"],
    "onnx_bytes": int(row["onnx_bytes"]),
    "int8_test_accuracy": v["int8_accuracy"],
    "shipped": "int8" if v["ship_int8"] else "fp32",
}
with open('artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
```

Cell 5 — download everything:

```python
import os, shutil
os.chdir('/content/checkmaize')
shutil.make_archive('/content/artifacts', 'zip', 'artifacts')
shutil.make_archive('/content/fixtures', 'zip', 'app/src/ml/__tests__/fixtures')
from google.colab import files
files.download('/content/artifacts.zip')
files.download('/content/fixtures.zip')
files.download('docs/onnx-contract.md')
```

- [ ] **Step 2: Commit the notebook**

```bash
git add colab/03_export.ipynb
git commit -m "feat: add export quantize verify colab notebook"
```

**Execution gate (user):** Unzip `artifacts.zip` into local `artifacts/` and `fixtures.zip` into `app/src/ml/__tests__/fixtures/`, and save `onnx-contract.md` into `docs/`. Expected verify output shows `ship_int8: true`.

---

### Task 13: App scaffold (Expo SDK 57 + onnxruntime fixes)

Creates the Expo app with pinned SDK 57, installs the mobile stack, applies the two verified onnxruntime-react-native fixes (delete `unimodule.json`; strip the Gradle `VersionNumber` block) via a postinstall script, configures jest, metro asset support for `.onnx`, and app.json.

**Files:**
- Create: `app/` (via create-expo-app), `app/scripts/fix-onnxruntime.js`, `app/metro.config.js`, modify `app/package.json`, modify `app/app.json`
- Test: verification steps below

**Interfaces:**
- Consumes: Node ≥ 22.13 (check first); internet for npm.
- Produces: the app skeleton + fixed onnxruntime package; the base for Tasks 14–19.

- [ ] **Step 1: Check Node version**

Run: `node --version`
Expected: `v22.13.0` or higher. If lower, upgrade Node first (SDK 57 requires it).

- [ ] **Step 2: Create the app**

Run: `npx create-expo-app@latest app --template blank-typescript --yes`
Expected: `app/` created with Expo SDK 57 (verify: `cd app && npx expo --version` prints an SDK 57 release; `app/package.json` contains `"expo": "~57.*"`).

- [ ] **Step 3: Install dependencies**

```bash
cd app
npx expo install expo-dev-client expo-camera expo-image-picker expo-image-manipulator expo-file-system expo-location expo-sqlite expo-sharing expo-asset react-native-screens react-native-safe-area-context
npm install onnxruntime-react-native@1.24.3 @react-navigation/native @react-navigation/native-stack @react-navigation/bottom-tabs jpeg-js
npm install -D jest-expo jest pngjs @types/jpeg-js @types/react
```

Expected: install completes; `npm ls onnxruntime-react-native` shows `1.24.3`.

- [ ] **Step 4: Write the postinstall fix script**

Create `app/scripts/fix-onnxruntime.js`:

```js
const fs = require('fs');
const path = require('path');

const pkgRoot = path.join(__dirname, '..', 'node_modules', 'onnxruntime-react-native');
const unimodule = path.join(pkgRoot, 'unimodule.json');
if (fs.existsSync(unimodule)) {
  fs.rmSync(unimodule);
  console.log('fix-onnxruntime: removed unimodule.json (unblocks Expo autolinking)');
}

const gradle = path.join(pkgRoot, 'android', 'build.gradle');
if (fs.existsSync(gradle)) {
  const contents = fs.readFileSync(gradle, 'utf8');
  const block = /  if \(VersionNumber\.parse\(REACT_NATIVE_VERSION\)[\s\S]*?\n  \}\n/;
  if (block.test(contents)) {
    fs.writeFileSync(gradle, contents.replace(block, ''));
    console.log('fix-onnxruntime: removed dead VersionNumber gradle block');
  }
}
```

- [ ] **Step 5: Wire postinstall + test script into `app/package.json`**

Add to `app/package.json`:

```json
"scripts": {
  "postinstall": "node scripts/fix-onnxruntime.js",
  "test": "jest"
},
"jest": {
  "preset": "jest-expo",
  "transformIgnorePatterns": [
    "node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|onnxruntime-react-native)"
  ]
}
```

(Keep the other generated scripts as-is.)

- [ ] **Step 6: Write `app/metro.config.js`**

```js
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('onnx');
module.exports = config;
```

- [ ] **Step 7: Update `app/app.json`**

Set `"name": "CheckMaize"`, `"slug": "checkmaize"`, add `"plugins": ["expo-dev-client"]`, and set identifiers (replace with your own if you have them):

```json
{
  "expo": {
    "name": "CheckMaize",
    "slug": "checkmaize",
    "version": "1.0.0",
    "orientation": "portrait",
    "userInterfaceStyle": "light",
    "plugins": ["expo-dev-client"],
    "android": {
      "package": "com.checkmaize.app"
    },
    "ios": {
      "bundleIdentifier": "com.checkmaize.app"
    }
  }
}
```

- [ ] **Step 8: Verify the onnxruntime fix**

```bash
cd app && rm -rf node_modules && npm install && node -e "const fs=require('fs');console.log('unimodule.json exists:',fs.existsSync('node_modules/onnxruntime-react-native/unimodule.json'))"
```

Expected: `unimodule.json exists: false`. Then verify autolinking registration:

```bash
cd app && npx expo-modules-autolinking react-native-config 2>/dev/null | grep -i onnx || echo "NOT REGISTERED"
```

Expected: output contains `onnxruntime-react-native` (or the grep prints the package entry). If not registered, STOP and report the output — do not proceed (this is the documented integration risk point).

- [ ] **Step 9: Verify the app boots in a dev build (Android)**

```bash
cd app && npx expo run:android
```

Expected: app builds and launches; Metro reports the app connected. (iOS equivalent: `npx expo run:ios` on a Mac.)

- [ ] **Step 10: Commit**

```bash
git add app
git commit -m "feat: scaffold expo sdk 57 app with onnxruntime autolink fixes"
```

---

### Task 14: Preprocessing module + parity test (`app/src/ml/`)

The JS mirror of the ONNX contract. Pure functions are fully unit-tested against the Python-generated fixture (`sample.png` + `reference_tensor.json` from Task 12); the native resize/file-read steps are isolated so the parity test never touches native modules.

**Files:**
- Create: `app/src/ml/contract.ts`
- Create: `app/src/ml/preprocess.ts`
- Test: `app/src/ml/__tests__/preprocess.test.ts`

**Interfaces:**
- Consumes: `app/src/ml/__tests__/fixtures/sample.png` + `reference_tensor.json` (Task 12).
- Produces:
  - `contract.ts`: `CONTRACT = { size: 224, mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225], inputName: 'input', outputName: 'output', inputShape: [1,3,224,224] }`, `ClassName` union type, `CLASS_ORDER: ClassName[]`, `Prediction { className, confidence }`.
  - `preprocess.ts`: `rgbaToCHWFloat32(rgba: Uint8Array, width, height, mean, std) -> Float32Array`, `resizeToContract(uri) -> Promise<string>`, `readFileBytes(uri) -> Promise<Uint8Array>`, `preprocessImage(uri) -> Promise<Float32Array>`.

- [ ] **Step 1: Ensure fixtures exist**

Run: `ls app/src/ml/__tests__/fixtures/sample.png app/src/ml/__tests__/fixtures/reference_tensor.json`
Expected: both files present (downloaded in the Task 12 gate).

- [ ] **Step 2: Write the failing test**

Create `app/src/ml/__tests__/preprocess.test.ts`:

```ts
import fs from 'fs';
import path from 'path';
import { decode } from 'pngjs';
import { rgbaToCHWFloat32 } from '../preprocess';
import { CONTRACT } from '../contract';

const fixtures = path.join(__dirname, 'fixtures');

test('fixture reference tensor has 32x32x3 values', () => {
  const ref = JSON.parse(fs.readFileSync(path.join(fixtures, 'reference_tensor.json'), 'utf8'));
  expect(ref).toHaveLength(32 * 32 * 3);
});

test('JS normalization matches the Python reference exactly', () => {
  const png = decode(fs.readFileSync(path.join(fixtures, 'sample.png')));
  const rgba = new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.length);
  const tensor = rgbaToCHWFloat32(rgba, png.width, png.height, CONTRACT.mean, CONTRACT.std);
  const ref = JSON.parse(fs.readFileSync(path.join(fixtures, 'reference_tensor.json'), 'utf8'));
  expect(tensor).toHaveLength(32 * 32 * 3);
  for (let i = 0; i < tensor.length; i++) {
    expect(Math.abs(tensor[i] - ref[i])).toBeLessThan(1e-5);
  }
});

test('CHW layout puts each channel in its own slab', () => {
  const rgba = new Uint8Array(4 * 2 * 2);
  rgba.set([255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255]);
  const t = rgbaToCHWFloat32(rgba, 2, 2, [0, 0, 0], [1, 1, 1]);
  expect(t[0]).toBeCloseTo(1.0, 5);
  expect(t[3]).toBeCloseTo(1.0, 5);
  expect(t[4]).toBeCloseTo(0.0, 5);
  expect(t[8]).toBeCloseTo(0.0, 5);
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd app && npx jest src/ml/__tests__/preprocess.test.ts`
Expected: FAIL — `Cannot find module '../contract' from 'preprocess.test.ts'` (or module resolution error for `../preprocess`).

- [ ] **Step 4: Write the implementation**

Create `app/src/ml/contract.ts`:

```ts
export const CONTRACT = {
  size: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  inputName: 'input',
  outputName: 'output',
  inputShape: [1, 3, 224, 224],
} as const;

export type ClassName = 'common_rust' | 'gray_leaf_spot' | 'northern_leaf_blight' | 'healthy';

export const CLASS_ORDER: ClassName[] = ['common_rust', 'gray_leaf_spot', 'northern_leaf_blight', 'healthy'];

export interface Prediction {
  className: ClassName;
  confidence: number;
}
```

Create `app/src/ml/preprocess.ts`:

```ts
import * as ImageManipulator from 'expo-image-manipulator';
import { decode as decodeJpeg } from 'jpeg-js';
import { CONTRACT } from './contract';

export function rgbaToCHWFloat32(
  rgba: Uint8Array,
  width: number,
  height: number,
  mean: readonly number[],
  std: readonly number[]
): Float32Array {
  const out = new Float32Array(3 * width * height);
  const pixels = width * height;
  for (let p = 0; p < pixels; p++) {
    for (let c = 0; c < 3; c++) {
      out[c * pixels + p] = (rgba[p * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

export async function resizeToContract(uri: string): Promise<string> {
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: CONTRACT.size, height: CONTRACT.size } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 1 }
  );
  return result.uri;
}

export async function readFileBytes(uri: string): Promise<Uint8Array> {
  const { File } = await import('expo-file-system');
  return new File(uri).bytes();
}

export async function preprocessImage(uri: string): Promise<Float32Array> {
  const resized = await resizeToContract(uri);
  const bytes = await readFileBytes(resized);
  const jpeg = decodeJpeg(bytes, { useTArray: true });
  return rgbaToCHWFloat32(jpeg.data, jpeg.width, jpeg.height, CONTRACT.mean, CONTRACT.std);
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd app && npx jest src/ml/__tests__/preprocess.test.ts`
Expected: PASS (3 tests). If jest cannot resolve `expo-image-manipulator` or `jpeg-js`, add this mock at the top of the test file:

```ts
jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: jest.fn(async () => ({ uri: 'mock.jpg' })),
  SaveFormat: { JPEG: 'jpeg' },
}));
```

- [ ] **Step 6: Commit**

```bash
git add app/src/ml
git commit -m "feat: add contract preprocessing module with python parity test"
```

---

### Task 15: ONNX classifier wrapper (`app/src/ml/onnx.ts`)

Wraps onnxruntime-react-native: loads the bundled model asset, runs inference on a preprocessed tensor, softmaxes logits, and returns sorted predictions. Designed so the logic is testable with a fake session (no native module needed in jest).

**Files:**
- Create: `app/src/ml/onnx.ts`
- Test: `app/src/ml/__tests__/onnx.test.ts`

**Interfaces:**
- Consumes: `contract.ts` (Task 14); `app/assets/model/model_int8.onnx` (added in Task 20).
- Produces: `MaizeClassifier` class with `static create(modelModule: number, sessionOptions?)`, `static fromSession(session)`, `classify(tensor: Float32Array) -> Promise<Prediction[]>`, `release()`.

- [ ] **Step 1: Write the failing test**

Create `app/src/ml/__tests__/onnx.test.ts`:

```ts
import type { InferenceSession } from 'onnxruntime-react-native';
import { MaizeClassifier } from '../onnx';

function fakeSession(logits: number[]) {
  return {
    run: jest.fn(async () => ({ output: { data: new Float32Array(logits) } })),
    release: jest.fn(async () => {}),
  } as unknown as InferenceSession;
}

test('classify returns softmax-sorted predictions', async () => {
  const session = fakeSession([0.1, 0.9, 0.2, 0.3]);
  const classifier = MaizeClassifier.fromSession(session);
  const tensor = new Float32Array(3 * 224 * 224);
  const preds = await classifier.classify(tensor);
  expect(session.run).toHaveBeenCalledWith({ input: expect.anything() });
  expect(preds).toHaveLength(4);
  expect(preds[0].className).toBe('gray_leaf_spot');
  expect(preds[0].confidence).toBeCloseTo(0.4008, 3);
  const confidences = preds.map((p) => p.confidence);
  expect([...confidences].sort((a, b) => b - a)).toEqual(confidences);
});

test('release delegates to the session', async () => {
  const session = fakeSession([0.25, 0.25, 0.25, 0.25]);
  const classifier = MaizeClassifier.fromSession(session);
  await classifier.release();
  expect(session.release).toHaveBeenCalled();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx jest src/ml/__tests__/onnx.test.ts`
Expected: FAIL — `Cannot find module '../onnx'`.

- [ ] **Step 3: Write the implementation**

Create `app/src/ml/onnx.ts`:

```ts
import { Asset } from 'expo-asset';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { CLASS_ORDER, CONTRACT } from './contract';
import type { ClassName, Prediction } from './contract';

export class MaizeClassifier {
  private constructor(private readonly session: InferenceSession) {}

  static async create(
    modelModule: number,
    sessionOptions?: InferenceSession.SessionOptions
  ): Promise<MaizeClassifier> {
    const asset = Asset.fromModule(modelModule);
    await asset.downloadAsync();
    const uri = (asset.localUri ?? asset.uri).replace(/^file:\/\//, '');
    const session = await InferenceSession.create(uri, {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all',
      ...sessionOptions,
    });
    return new MaizeClassifier(session);
  }

  static fromSession(session: InferenceSession): MaizeClassifier {
    return new MaizeClassifier(session);
  }

  async classify(tensor: Float32Array): Promise<Prediction[]> {
    const input = new Tensor('float32', tensor, CONTRACT.inputShape);
    const result = await this.session.run({ [CONTRACT.inputName]: input });
    const logits = Array.from(result[CONTRACT.outputName].data as Float32Array);
    const max = Math.max(...logits);
    const exp = logits.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp
      .map((p, i) => ({ className: CLASS_ORDER[i] as ClassName, confidence: p / sum }))
      .sort((a, b) => b.confidence - a.confidence);
  }

  async release(): Promise<void> {
    await this.session.release();
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx jest src/ml/__tests__/onnx.test.ts`
Expected: PASS (2 tests). If `expo-asset` import fails in jest, add `jest.mock('expo-asset', () => ({ Asset: { fromModule: jest.fn(() => ({ downloadAsync: jest.fn(async () => {}), localUri: '/tmp/m.onnx', uri: '/tmp/m.onnx' })) } }));` to the test file.

- [ ] **Step 5: Commit**

```bash
git add app/src/ml/onnx.ts app/src/ml/__tests__/onnx.test.ts
git commit -m "feat: add onnx classifier wrapper with fake-session tests"
```

---

### Task 16: Disease knowledge base (`app/src/data/diseases.ts`)

The RQ1 distilate: static, bundled per-class guidance (visual hallmarks + practical management), plus the low-confidence threshold used by the result gate.

**Files:**
- Create: `app/src/data/diseases.ts`
- Test: `app/src/data/__tests__/diseases.test.ts`

**Interfaces:**
- Consumes: `contract.ts` (Task 14).
- Produces: `LOW_CONFIDENCE_THRESHOLD: number`, `DISEASES: Record<ClassName, DiseaseInfo>`, `getDiseaseInfo(className) -> DiseaseInfo`. `DiseaseInfo = { id, name, causalAgent, hallmarks: string[], management: string[] }`.

- [ ] **Step 1: Write the failing test**

Create `app/src/data/__tests__/diseases.test.ts`:

```ts
import { CLASS_ORDER } from '../../ml/contract';
import { DISEASES, LOW_CONFIDENCE_THRESHOLD, getDiseaseInfo } from '../diseases';

test('every class in the contract has an info entry', () => {
  for (const cls of CLASS_ORDER) {
    expect(DISEASES[cls]).toBeDefined();
    expect(DISEASES[cls].id).toBe(cls);
  }
});

test('info entries are complete and non-empty', () => {
  for (const cls of CLASS_ORDER) {
    const info = DISEASES[cls];
    expect(info.name.length).toBeGreaterThan(0);
    expect(info.causalAgent.length).toBeGreaterThan(0);
    expect(info.hallmarks.length).toBeGreaterThan(0);
    expect(info.management.length).toBeGreaterThan(0);
  }
});

test('threshold is a probability', () => {
  expect(LOW_CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
  expect(LOW_CONFIDENCE_THRESHOLD).toBeLessThan(1);
});

test('getDiseaseInfo returns the healthy entry', () => {
  expect(getDiseaseInfo('healthy').name).toBe('Healthy Maize Leaf');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx jest src/data/__tests__/diseases.test.ts`
Expected: FAIL — `Cannot find module '../diseases'`.

- [ ] **Step 3: Write the implementation**

Create `app/src/data/diseases.ts`:

```ts
import { CLASS_ORDER } from '../ml/contract';
import type { ClassName } from '../ml/contract';

export const LOW_CONFIDENCE_THRESHOLD = 0.6;

export interface DiseaseInfo {
  id: ClassName;
  name: string;
  causalAgent: string;
  hallmarks: string[];
  management: string[];
}

export const DISEASES: Record<ClassName, DiseaseInfo> = {
  common_rust: {
    id: 'common_rust',
    name: 'Common Rust',
    causalAgent: 'Puccinia sorghi (fungus)',
    hallmarks: [
      'Dusty reddish-brown pustules on both leaf surfaces',
      'Pustules rupture to release rust-colored spores',
      'Usually appears after tasseling in wet weather',
    ],
    management: [
      'Grow resistant hybrids',
      'Rotate away from maize for a season',
      'Fungicide only if severe before tasseling',
    ],
  },
  gray_leaf_spot: {
    id: 'gray_leaf_spot',
    name: 'Gray Leaf Spot',
    causalAgent: 'Cercospora zeina / Cercospora zeae-maydis (fungus)',
    hallmarks: [
      'Rectangular tan-to-gray lesions running parallel to leaf veins',
      'Lesions narrow and bounded by veins',
      'Leaves look blighted where lesions coalesce',
    ],
    management: [
      'Use resistant varieties',
      'Remove or bury maize residue after harvest',
      'Strobilurin or triazole fungicide if disease starts before tasseling',
    ],
  },
  northern_leaf_blight: {
    id: 'northern_leaf_blight',
    name: 'Northern Leaf Blight',
    causalAgent: 'Exserohilum turcicum (fungus)',
    hallmarks: [
      'Large cigar-shaped grey-green to tan lesions',
      'Lesions 2.5-30 cm long, tapering at both ends',
      'Starts on lower leaves and spreads upward',
    ],
    management: [
      'Plant resistant hybrids',
      'Crop rotation with non-cereal crops',
      'Fungicide at tasseling if upper leaves are infected',
    ],
  },
  healthy: {
    id: 'healthy',
    name: 'Healthy Maize Leaf',
    causalAgent: 'None detected',
    hallmarks: ['Uniform green color', 'No lesions, pustules or discoloration'],
    management: ['No action needed', 'Keep scouting; recheck after wet weather'],
  },
};

export function getDiseaseInfo(className: ClassName): DiseaseInfo {
  return DISEASES[className];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx jest src/data/__tests__/diseases.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/data
git commit -m "feat: add disease knowledge base and confidence threshold"
```

---

### Task 17: Scan log + geotagging (`app/src/db/scans.ts`)

SQLite-backed local scan history with optional GPS coordinates; consent is enforced by the caller (only coordinates recorded when permission is granted). All DB logic unit-tested against an in-memory mock.

**Files:**
- Create: `app/src/db/scans.ts`
- Test: `app/src/db/__tests__/scans.test.ts`

**Interfaces:**
- Produces: `ScanRecord { id?, createdAt, imageUri, prediction: ClassName, confidence, latitude: number|null, longitude: number|null }`; `initDb() -> Promise<SQLiteDatabase>`; `saveScan(record) -> Promise<ScanRecord>`; `getScans() -> Promise<ScanRecord[]>` (newest first); `deleteScan(id) -> Promise<void>`.

- [ ] **Step 1: Write the failing test**

Create `app/src/db/__tests__/scans.test.ts`:

```ts
jest.mock('expo-sqlite', () => {
  let nextId = 0;
  const rows: any[] = [];
  const db = {
    execAsync: jest.fn(async () => {}),
    runAsync: jest.fn(async (_sql: string, ...args: any[]) => {
      nextId += 1;
      rows.push({ id: nextId, created_at: args[0], image_uri: args[1], prediction: args[2], confidence: args[3], latitude: args[4], longitude: args[5] });
      return { lastInsertRowId: nextId };
    }),
    getAllAsync: jest.fn(async () => rows),
  };
  return { openDatabaseAsync: jest.fn(async () => db) };
});

import { saveScan, getScans, deleteScan } from '../scans';

test('saved scans round-trip with geotag nulls', async () => {
  const saved = await saveScan({
    createdAt: '2026-08-04T10:00:00Z',
    imageUri: 'file:///a.jpg',
    prediction: 'northern_leaf_blight',
    confidence: 0.91,
    latitude: null,
    longitude: null,
  });
  expect(saved.id).toBeGreaterThan(0);
  const scans = await getScans();
  expect(scans).toHaveLength(1);
  expect(scans[0]).toMatchObject({
    createdAt: '2026-08-04T10:00:00Z',
    prediction: 'northern_leaf_blight',
    confidence: 0.91,
    latitude: null,
    longitude: null,
  });
});

test('newest scan comes first and delete removes it', async () => {
  await saveScan({ createdAt: '2026-08-04T09:00:00Z', imageUri: 'file:///a.jpg', prediction: 'common_rust', confidence: 0.8, latitude: 5.6, longitude: -0.2 });
  await saveScan({ createdAt: '2026-08-04T11:00:00Z', imageUri: 'file:///b.jpg', prediction: 'healthy', confidence: 0.99, latitude: null, longitude: null });
  const scans = await getScans();
  expect(scans[0].prediction).toBe('healthy');
  expect(scans[1].latitude).toBe(5.6);
  await deleteScan(scans[0].id!);
  const after = await getScans();
  expect(after).toHaveLength(1);
  expect(after[0].prediction).toBe('common_rust');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx jest src/db/__tests__/scans.test.ts`
Expected: FAIL — `Cannot find module '../scans'`.

- [ ] **Step 3: Write the implementation**

Create `app/src/db/scans.ts`:

```ts
import * as SQLite from 'expo-sqlite';
import type { ClassName } from '../ml/contract';

export interface ScanRecord {
  id?: number;
  createdAt: string;
  imageUri: string;
  prediction: ClassName;
  confidence: number;
  latitude: number | null;
  longitude: number | null;
}

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

export function initDb(): Promise<SQLite.SQLiteDatabase> {
  if (!dbPromise) {
    dbPromise = SQLite.openDatabaseAsync('checkmaize.db').then(async (db) => {
      await db.execAsync(
        `CREATE TABLE IF NOT EXISTS scans (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          image_uri TEXT NOT NULL,
          prediction TEXT NOT NULL,
          confidence REAL NOT NULL,
          latitude REAL,
          longitude REAL
        )`
      );
      return db;
    });
  }
  return dbPromise;
}

export async function saveScan(record: ScanRecord): Promise<ScanRecord> {
  const db = await initDb();
  const result = await db.runAsync(
    'INSERT INTO scans (created_at, image_uri, prediction, confidence, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)',
    record.createdAt,
    record.imageUri,
    record.prediction,
    record.confidence,
    record.latitude,
    record.longitude
  );
  return { ...record, id: result.lastInsertRowId };
}

export async function getScans(): Promise<ScanRecord[]> {
  const db = await initDb();
  const rows = await db.getAllAsync('SELECT * FROM scans ORDER BY created_at DESC');
  return rows.map((r) => ({
    id: r.id as number,
    createdAt: r.created_at as string,
    imageUri: r.image_uri as string,
    prediction: r.prediction as ClassName,
    confidence: r.confidence as number,
    latitude: r.latitude as number | null,
    longitude: r.longitude as number | null,
  }));
}

export async function deleteScan(id: number): Promise<void> {
  const db = await initDb();
  await db.runAsync('DELETE FROM scans WHERE id = ?', id);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx jest src/db/__tests__/scans.test.ts`
Expected: PASS (2 tests). If the mock doesn't type-check under jest-expo, cast the mocked db rows (`r as any`) in `getScans` — note this in the commit if needed.

- [ ] **Step 5: Commit**

```bash
git add app/src/db
git commit -m "feat: add sqlite scan log with geotag support"
```

---

### Task 18: Model context + Capture/Result screens

The classifier is loaded once in a React context; the Capture screen drives camera/gallery → preprocess → classify → save → navigate, and applies the low-confidence gate ("unclear, re-capture or consult an extension agent") before ever showing a diagnosis. GPS is only recorded when the user grants foreground location permission.

**Files:**
- Create: `app/src/ml/ModelContext.tsx`
- Create: `app/src/screens/CaptureScreen.tsx`
- Create: `app/src/screens/ResultScreen.tsx`

**Interfaces:**
- Consumes: `preprocessImage` (Task 14), `MaizeClassifier` (Task 15), `saveScan` (Task 17), `LOW_CONFIDENCE_THRESHOLD` (Task 16), `app/assets/model/model_int8.onnx` (Task 20).
- Produces: `ModelProvider` + `useClassifier()` hook exposing `{ ready, classify(uri), metrics }`; navigation params `{ imageUri: string; prediction: ClassName; confidence: number }` for the `Result` route.

- [ ] **Step 1: Write the model context**

Create `app/src/ml/ModelContext.tsx`:

```tsx
import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { MaizeClassifier } from './onnx';
import { preprocessImage } from './preprocess';
import type { Prediction } from './contract';

const modelModule = require('../../assets/model/model_int8.onnx') as number;

export interface ModelMetrics {
  model: string;
  test_accuracy: number;
  macro_f1: number;
  onnx_bytes: number;
  int8_test_accuracy: number;
  shipped: string;
}

interface ModelContextValue {
  ready: boolean;
  classify: (uri: string) => Promise<Prediction[]>;
  metrics: ModelMetrics | null;
}

const ModelContext = createContext<ModelContextValue | null>(null);

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [classifier, setClassifier] = useState<MaizeClassifier | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [classifierInstance, metricsData] = await Promise.all([
          MaizeClassifier.create(modelModule),
          require('../../assets/model/metrics.json') as ModelMetrics,
        ]);
        if (!cancelled) {
          setClassifier(classifierInstance);
          setMetrics(metricsData);
          setReady(true);
        }
      } catch (error) {
        console.error('model load failed', error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<ModelContextValue>(
    () => ({
      ready,
      classify: async (uri: string) => {
        if (!classifier) {
          throw new Error('classifier not ready');
        }
        const tensor = await preprocessImage(uri);
        return classifier.classify(tensor);
      },
      metrics,
    }),
    [ready, classifier, metrics]
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useClassifier(): ModelContextValue {
  const value = useContext(ModelContext);
  if (!value) {
    throw new Error('useClassifier must be used within ModelProvider');
  }
  return value;
}
```

- [ ] **Step 2: Write the Capture screen**

Create `app/src/screens/CaptureScreen.tsx`:

```tsx
import React, { useEffect, useRef, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { CameraView } from 'expo-camera';
import * as Camera from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useClassifier } from '../ml/ModelContext';
import { saveScan } from '../db/scans';
import { LOW_CONFIDENCE_THRESHOLD } from '../data/diseases';
import type { RootStackParamList } from '../../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Tabs'>;

export default function CaptureScreen({ navigation }: Props) {
  const cameraRef = useRef<CameraView>(null);
  const { ready, classify } = useClassifier();
  const [permission, setPermission] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [unclear, setUnclear] = useState(false);

  useEffect(() => {
    Camera.requestCameraPermissionsAsync().then((r) => setPermission(r.granted));
  }, []);

  const runScan = async (uri: string) => {
    if (!ready) {
      Alert.alert('Model still loading', 'Please wait a moment and try again.');
      return;
    }
    setBusy(true);
    setUnclear(false);
    try {
      const predictions = await classify(uri);
      const top = predictions[0];
      let latitude: number | null = null;
      let longitude: number | null = null;
      const perm = await Location.requestForegroundPermissionsAsync();
      if (perm.granted) {
        const pos = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
        latitude = pos.coords.latitude;
        longitude = pos.coords.longitude;
      }
      await saveScan({
        createdAt: new Date().toISOString(),
        imageUri: uri,
        prediction: top.className,
        confidence: top.confidence,
        latitude,
        longitude,
      });
      if (top.confidence >= LOW_CONFIDENCE_THRESHOLD) {
        navigation.navigate('Result', { imageUri: uri, prediction: top.className, confidence: top.confidence });
      } else {
        setUnclear(true);
      }
    } catch (error) {
      Alert.alert('Scan failed', String(error));
    } finally {
      setBusy(false);
    }
  };

  const onCapture = async () => {
    const photo = await cameraRef.current?.takePictureAsync({ quality: 1 });
    if (photo) {
      await runScan(photo.uri);
    }
  };

  const onPick = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 1 });
    if (!result.canceled) {
      await runScan(result.assets[0].uri);
    }
  };

  if (permission === null) {
    return <View style={styles.center}><Text>Requesting camera permission...</Text></View>;
  }
  if (permission === false) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Camera access is required</Text>
        <Pressable style={styles.button} onPress={onPick}>
          <Text style={styles.buttonText}>Choose photo from gallery instead</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
      <Text style={styles.tip}>Hold the camera close to a single symptomatic leaf, filling the frame.</Text>
      {unclear && (
        <View style={styles.unclearBox}>
          <Text style={styles.unclearText}>Unclear photo - re-capture with better light, or ask your extension agent.</Text>
        </View>
      )}
      <View style={styles.row}>
        <Pressable style={styles.button} onPress={onPick} disabled={busy}>
          <Text style={styles.buttonText}>Gallery</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.shutter]} onPress={onCapture} disabled={busy}>
          <Text style={styles.buttonText}>{busy ? 'Scanning...' : 'Capture'}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, padding: 24 },
  title: { fontSize: 18, fontWeight: '600' },
  tip: { color: '#fff', textAlign: 'center', padding: 8, backgroundColor: 'rgba(0,0,0,0.6)' },
  unclearBox: { backgroundColor: '#fde68a', padding: 10 },
  unclearText: { color: '#92400e', textAlign: 'center' },
  row: { flexDirection: 'row', gap: 12, padding: 16, justifyContent: 'center' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 24 },
  shutter: { backgroundColor: '#166534' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
```

- [ ] **Step 3: Write the Result screen**

Create `app/src/screens/ResultScreen.tsx`:

```tsx
import React from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { getDiseaseInfo } from '../data/diseases';
import type { RootStackParamList } from '../../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Result'>;

export default function ResultScreen({ navigation, route }: Props) {
  const { imageUri, prediction, confidence } = route.params;
  const info = getDiseaseInfo(prediction);
  const percent = Math.round(confidence * 100);

  return (
    <View style={styles.container}>
      <Image source={{ uri: imageUri }} style={styles.image} />
      <Text style={styles.name}>{info.name}</Text>
      <View style={styles.bar}>
        <View style={[styles.barFill, { width: `${percent}%` }]} />
      </View>
      <Text style={styles.confidence}>{percent}% confidence</Text>
      <Text style={styles.hallmarks}>{info.hallmarks[0]}</Text>
      <Pressable style={styles.button} onPress={() => navigation.navigate('Info', { className: prediction })}>
        <Text style={styles.buttonText}>What to do about it</Text>
      </Pressable>
      <Pressable style={[styles.button, styles.secondary]} onPress={() => navigation.goBack()}>
        <Text style={styles.buttonText}>Scan another leaf</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', padding: 24, gap: 12 },
  image: { width: 220, height: 220, borderRadius: 12 },
  name: { fontSize: 24, fontWeight: '700' },
  bar: { width: '100%', height: 10, backgroundColor: '#e5e7eb', borderRadius: 5, overflow: 'hidden' },
  barFill: { height: '100%', backgroundColor: '#15803d' },
  confidence: { color: '#4b5563' },
  hallmarks: { color: '#374151', textAlign: 'center' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 32 },
  secondary: { backgroundColor: '#166534' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
```

- [ ] **Step 4: Run the type check**

Run: `cd app && npx tsc --noEmit`
Expected: one type error only — `RootStackParamList` is not yet exported from `App.tsx` (it's created in Task 19). If other errors appear, fix them before committing.

- [ ] **Step 5: Commit**

```bash
git add app/src/ml/ModelContext.tsx app/src/screens
git commit -m "feat: add model context and capture/result screens"
```

---

### Task 19: Remaining screens + navigation root (`App.tsx`)

Info, Log, Contribute and About screens, plus the navigation root that wires tabs and stack routes and exports the `RootStackParamList` type used by the screens.

**Files:**
- Create: `app/src/screens/InfoScreen.tsx`
- Create: `app/src/screens/LogScreen.tsx`
- Create: `app/src/screens/ContributeScreen.tsx`
- Create: `app/src/screens/AboutScreen.tsx`
- Modify: `app/App.tsx`

**Interfaces:**
- Consumes: `DISEASES`/`getDiseaseInfo` (Task 16), `getScans`/`deleteScan` (Task 17), `useClassifier` (Task 18).
- Produces: `RootStackParamList = { Tabs: undefined; Result: { imageUri: string; prediction: ClassName; confidence: number }; Info: { className: ClassName } }` exported from `App.tsx`.

- [ ] **Step 1: Write the Info screen**

Create `app/src/screens/InfoScreen.tsx`:

```tsx
import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { DISEASES } from '../data/diseases';
import type { RootStackParamList } from '../../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Info'>;

export default function InfoScreen({ route }: Props) {
  const info = DISEASES[route.params.className];
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.name}>{info.name}</Text>
      <Text style={styles.agent}>Cause: {info.causalAgent}</Text>
      <Text style={styles.section}>How to recognise it</Text>
      {info.hallmarks.map((h, i) => (
        <Text key={i} style={styles.item}>{'\u2022'} {h}</Text>
      ))}
      <Text style={styles.section}>Recommended management</Text>
      {info.management.map((m, i) => (
        <Text key={i} style={styles.item}>{'\u2022'} {m}</Text>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, gap: 8 },
  name: { fontSize: 22, fontWeight: '700' },
  agent: { color: '#4b5563' },
  section: { fontSize: 16, fontWeight: '600', marginTop: 12 },
  item: { color: '#374151', lineHeight: 22 },
});
```

- [ ] **Step 2: Write the Log screen**

Create `app/src/screens/LogScreen.tsx`:

```tsx
import React, { useCallback, useState } from 'react';
import { FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { getScans, deleteScan } from '../db/scans';
import type { ScanRecord } from '../db/scans';
import { getDiseaseInfo } from '../data/diseases';

export default function LogScreen() {
  const [scans, setScans] = useState<ScanRecord[]>([]);

  useFocusEffect(
    useCallback(() => {
      getScans().then(setScans);
    }, [])
  );

  const remove = async (id: number) => {
    await deleteScan(id);
    setScans(await getScans());
  };

  return (
    <FlatList
      data={scans}
      keyExtractor={(s) => String(s.id)}
      contentContainerStyle={styles.container}
      renderItem={({ item }) => (
        <View style={styles.row}>
          <Image source={{ uri: item.imageUri }} style={styles.thumb} />
          <View style={styles.info}>
            <Text style={styles.name}>{getDiseaseInfo(item.prediction).name}</Text>
            <Text style={styles.meta}>
              {new Date(item.createdAt).toLocaleString()} - {Math.round(item.confidence * 100)}%
            </Text>
            <Text style={styles.meta}>
              {item.latitude != null ? `${item.latitude.toFixed(4)}, ${item.longitude!.toFixed(4)}` : 'no location'}
            </Text>
          </View>
          <Pressable onPress={() => remove(item.id!)}>
            <Text style={styles.delete}>Delete</Text>
          </Pressable>
        </View>
      )}
      ListEmptyComponent={<Text style={styles.empty}>No scans yet</Text>}
    />
  );
}

const styles = StyleSheet.create({
  container: { padding: 12, gap: 8 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#fff', borderRadius: 8, padding: 8 },
  thumb: { width: 56, height: 56, borderRadius: 6 },
  info: { flex: 1 },
  name: { fontWeight: '600' },
  meta: { color: '#6b7280', fontSize: 12 },
  delete: { color: '#dc2626' },
  empty: { textAlign: 'center', marginTop: 40, color: '#6b7280' },
});
```

- [ ] **Step 3: Write the Contribute screen**

Create `app/src/screens/ContributeScreen.tsx`:

```tsx
import React, { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';

export default function ContributeScreen() {
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  const onContribute = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 1 });
    if (result.canceled) {
      return;
    }
    const uri = result.assets[0].uri;
    const source = new File(uri);
    const bytes = await source.bytes();
    const dir = new File(Paths.document, 'contributions/');
    if (!dir.exists) {
      dir.create({ intermediates: true });
    }
    const name = `contribution_${Date.now()}.jpg`;
    const dest = new File(dir, name);
    dest.write(bytes);
    setLastSaved(dest.uri);
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(dest.uri, {
        mimeType: 'image/jpeg',
        dialogTitle: 'Share this collection photo',
      });
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Help build the Ghana field dataset</Text>
      <Text style={styles.body}>
        Choose a leaf photo (ideally in its field background). It is saved to the app folder and you can
        share it to the researcher (WhatsApp, email, Drive). No labels are needed - the researcher labels later.
      </Text>
      {lastSaved && <Text style={styles.saved}>Saved: {lastSaved}</Text>}
      <Pressable style={styles.button} onPress={onContribute}>
        <Text style={styles.buttonText}>Pick a photo to contribute</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12, backgroundColor: '#fff' },
  title: { fontSize: 20, fontWeight: '700' },
  body: { color: '#374151', lineHeight: 22 },
  saved: { color: '#15803d' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
```

- [ ] **Step 4: Write the About screen**

Create `app/src/screens/AboutScreen.tsx`:

```tsx
import React, { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useClassifier } from '../ml/ModelContext';

export default function AboutScreen() {
  const { metrics, classify } = useClassifier();
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const timeInference = async () => {
    const start = Date.now();
    await classify('latency-probe');
    setLatencyMs(Date.now() - start);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>About the model</Text>
      {metrics ? (
        <>
          <Text style={styles.line}>Architecture: {metrics.model}</Text>
          <Text style={styles.line}>Test accuracy: {(metrics.test_accuracy * 100).toFixed(1)}%</Text>
          <Text style={styles.line}>Macro F1: {metrics.macro_f1.toFixed(3)}</Text>
          <Text style={styles.line}>Int8 test accuracy: {(metrics.int8_test_accuracy * 100).toFixed(1)}%</Text>
          <Text style={styles.line}>Shipped: {metrics.shipped} ({Math.round(metrics.onnx_bytes / 1024)} KB)</Text>
        </>
      ) : (
        <Text style={styles.line}>Model metrics not bundled.</Text>
      )}
      <Pressable style={styles.button} onPress={timeInference}>
        <Text style={styles.buttonText}>Time one on-device inference</Text>
      </Pressable>
      {latencyMs != null && <Text style={styles.line}>Inference took {latencyMs} ms</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 10, backgroundColor: '#fff' },
  title: { fontSize: 20, fontWeight: '700' },
  line: { color: '#374151' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, alignItems: 'center', marginTop: 12 },
  buttonText: { color: '#fff', fontWeight: '600' },
});
```

- [ ] **Step 5: Rewrite `app/App.tsx`**

```tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { ModelProvider } from './src/ml/ModelContext';
import CaptureScreen from './src/screens/CaptureScreen';
import ResultScreen from './src/screens/ResultScreen';
import InfoScreen from './src/screens/InfoScreen';
import LogScreen from './src/screens/LogScreen';
import ContributeScreen from './src/screens/ContributeScreen';
import AboutScreen from './src/screens/AboutScreen';
import type { ClassName } from './src/ml/contract';

export type RootStackParamList = {
  Tabs: undefined;
  Result: { imageUri: string; prediction: ClassName; confidence: number };
  Info: { className: ClassName };
};

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tabs = createBottomTabNavigator();

function TabsNavigator() {
  return (
    <Tabs.Navigator>
      <Tabs.Screen name="Scan" component={CaptureScreen} />
      <Tabs.Screen name="History" component={LogScreen} />
      <Tabs.Screen name="Contribute" component={ContributeScreen} />
      <Tabs.Screen name="About" component={AboutScreen} />
    </Tabs.Navigator>
  );
}

export default function App() {
  return (
    <ModelProvider>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen name="Tabs" component={TabsNavigator} options={{ headerShown: false }} />
          <Stack.Screen name="Result" component={ResultScreen} />
          <Stack.Screen name="Info" component={InfoScreen} />
        </Stack.Navigator>
      </NavigationContainer>
      <StatusBar style="light" />
    </ModelProvider>
  );
}
```

- [ ] **Step 6: Run the type check**

Run: `cd app && npx tsc --noEmit`
Expected: PASS with no errors.

- [ ] **Step 7: Run all app tests**

Run: `cd app && npx jest`
Expected: PASS — 11 tests (3 preprocess + 2 onnx + 4 diseases + 2 scans).

- [ ] **Step 8: Commit**

```bash
git add app/App.tsx app/src/screens
git commit -m "feat: add info log contribute about screens and navigation"
```

---

### Task 20: Model bundling, EAS builds and handover docs

Bundles the shipped model into the app (the one committed binary), configures EAS, produces the installable APK/IPA, and writes the handover docs: `docs/dataset-notes.md`, `docs/pilot-protocol.md`, `docs/colab-workflow.md`.

**Files:**
- Create: `app/assets/model/` (model_int8.onnx, labels.json, metrics.json from Task 12 artifacts)
- Create: `eas.json` (in `app/`)
- Create: `docs/dataset-notes.md`
- Create: `docs/pilot-protocol.md`
- Create: `docs/colab-workflow.md`
- Modify: `app/README.md`

**Interfaces:**
- Consumes: `artifacts/model_int8.onnx` + `artifacts/labels.json` + `artifacts/metrics.json` + `docs/onnx-contract.md` (Task 12 gate).
- Produces: installable Android APK and iOS build; handover documentation for the academic writer and field pilot.

- [ ] **Step 1: Bundle the model assets**

```bash
mkdir -p app/assets/model
cp artifacts/model_int8.onnx app/assets/model/model_int8.onnx
cp artifacts/labels.json app/assets/model/labels.json
cp artifacts/metrics.json app/assets/model/metrics.json
ls -la app/assets/model
```

Expected: three files present; `model_int8.onnx` should be ~3–5 MB. If the verify gate chose fp32, copy `artifacts/runs/{winner}/model.onnx` as `model.onnx` instead and update `ModelContext.tsx`'s `require` path accordingly.

- [ ] **Step 2: Sanity-check the bundled model loads in dev**

Run: `cd app && npx expo run:android` and on the Scan tab confirm the app boots without the `Cannot read property 'install' of null` error. Then capture one photo and confirm a result screen appears (or the "unclear" box for low confidence).

- [ ] **Step 3: Write `app/eas.json`**

```json
{
  "cli": {
    "version": ">= 5.0.0"
  },
  "build": {
    "preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "autoIncrement": true
    }
  }
}
```

- [ ] **Step 4: Build the Android APK**

Run: `cd app && npx eas-cli@latest login` (interactive; user's Expo account), then `cd app && npx eas-cli@latest build -p android --profile preview`
Expected: EAS builds the app and prints an installable `.apk` download link. Download it and install on a test device (Android 7+).

- [ ] **Step 5: Build for iOS**

Run: `cd app && npx eas-cli@latest build -p ios --profile preview` (requires an Apple Developer account; follow the EAS prompts to register credentials)
Expected: an installable build (TestFlight or ad-hoc IPA per account settings). If the iOS build fails at model loading (known upstream issue #27062), confirm the asset-copy path in `MaizeClassifier.create` and retry; document the outcome in `docs/pilot-protocol.md` as a risk note.

- [ ] **Step 6: Write `docs/dataset-notes.md`**

```markdown
# Dataset Notes

## Sources

| Source | Origin | Classes used | Images |
|---|---|---|---|
| PlantVillage (via Hugging Face `mohanty/PlantVillage`) | Controlled/lab imagery | common_rust, gray_leaf_spot, northern_leaf_blight, healthy | ~4,000 maize images |
| CCMT `Dataset for Crop Pest and Disease Detection` (Mendeley DOI 10.17632/bwh3zbpkpv.1) | Real field farms in Ghana (University of Energy and Natural Resources, Sunyani); expert-labelled by plant virologists | northern_leaf_blight (Leaf blight), gray_leaf_spot (Leaf spot), healthy | ~2,470 maize images (subset of 5,389 raw) |

## Class mapping

- CCMT `Leaf blight` -> northern_leaf_blight; CCMT `Leaf spot` -> gray_leaf_spot; CCMT `Healthy` -> healthy.
- PlantVillage folder names map 1:1 to the catalog.
- CCMT classes not used (out of the 4-class catalog): fall armyworm, grasshopper, leaf beetle, streak virus.

## Known limitation (RQ2 honesty)

CCMT contains no Common Rust imagery, so the Ghanaian-field test set covers 3 classes
(northern_leaf_blight, gray_leaf_spot, healthy). Common Rust is evaluated on PlantVillage
test imagery. Local collection (app Contribute flow) is the planned path to close this gap.

## Splitting methodology

- Leakage prevention: images of the same physical leaf (`leaf_id`) never cross train/test.
- Main experiment: train = PlantVillage + 70% of Ghana, val = 10%, test = Ghana-only 20% holdout.
- Domain-shift experiment: train = PlantVillage only; test = ALL Ghana images.
- Seed 42; manifests are committed and reproducible.

## Provenance

- PlantVillage: Mohanty et al. (2016), Frontiers in Plant Science.
- CCMT: "Dataset for crop pest and disease detection", Mendeley Data v1, 2023,
  DOI 10.17632/bwh3zbpkpv.1.
```

- [ ] **Step 7: Write `docs/pilot-protocol.md`**

```markdown
# Field Pilot Protocol (Ghana)

## Objective

Estimate real-world accuracy of the installed app against expert labelling, and
collect new Ghana field photos for dataset growth.

## Setup

1. Install the preview APK on 5-10 farmer/extension-agent phones (Android 7+).
2. One trained enumerator (or an extension agent) accompanies users for the first session.
3. Print this sheet for reference; the app works fully offline.

## Protocol

1. Each participant captures 10-20 leaves (different plants, fields, times of day).
2. For every scan, an expert (plant virologist/extension officer) independently records
   their own label for the same leaf. Labels: common_rust, gray_leaf_spot,
   northern_leaf_blight, healthy, other.
3. Record: app prediction + confidence (from the History tab), expert label, location.

## Analysis

- Build a comparison table app-vs-expert; report accuracy and confusion matrix.
- Flag every disagreement; inspect the image to decide whether the app or the expert
  was right (or both wrong).
- Add "expert-corrected" images to the local collection -> `data/raw/local/` with
  `leaf_id = <expertid>_<plantid>`, re-run the data tasks, and optionally fine-tune.

## Feedback loop

Disagreements with good image quality are the highest-value training data. Send them
(Contribute tab -> share) to the researcher with the expert label.
```

- [ ] **Step 8: Write `docs/colab-workflow.md`**

```markdown
# Colab Workflow

All Python runs on Google Colab (free). The repo is the source of truth; notebooks are drivers.

## Prerequisites per session

- Push the repo to GitHub (or upload a zip to `/content/checkmaize.zip`).
- Notebook 01: download PlantVillage via HF (auto), upload `Raw Data.zip` (CCMT,
  Mendeley DOI 10.17632/bwh3zbpkpv.1) via the files pane.
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
```

- [ ] **Step 9: Update `app/README.md`**

Replace the generated README content with:

```markdown
# CheckMaize App

On-device maize leaf disease classifier for Ghana. Expo SDK 57 + onnxruntime-react-native.

## Model

The bundled int8 ONNX model (see `assets/model/metrics.json`) classifies
common_rust / gray_leaf_spot / northern_leaf_blight / healthy at 224x224 RGB.
The preprocessing contract is enforced by the parity test in `src/ml/__tests__/`.

## Development

- Node >= 22.13. `npm install` runs the onnxruntime autolink fix automatically.
- `npx expo run:android` (or `run:ios`) for a dev build; `npm test` for jest.

## Building installables

- Android APK: `npx eas-cli@latest build -p android --profile preview`
- iOS: `npx eas-cli@latest build -p ios --profile preview` (needs an Apple Developer account)

## Contributing field photos

Use the Contribute tab; photos are saved to the app folder and shared via the OS share sheet.
```

- [ ] **Step 10: Final commit**

```bash
git add app/assets/model app/eas.json app/README.md docs
git commit -m "feat: bundle model, add eas builds and handover docs"
```

**Final gate:** `cd app && npx jest` → all pass; `npx tsc --noEmit` → clean; Android APK installs and scans a leaf end-to-end offline.

---

## Self-Review

- **Spec coverage:** data pipeline (Tasks 2–4), two evaluation axes (Task 3 tests), RQ3 5-candidate benchmark (Tasks 6–8), export/quantize/parity gates (Tasks 9–11), app with all six screens + contribute flow + geotagged log + confidence gate (Tasks 13–19), EAS builds + pilot protocol + dataset notes + local-collection kit (Task 20). Domain-shift manifest and within-source Ghana test both produced by Task 3. All spec sections map to tasks.
- **Placeholder scan:** no TBD/TODO. The only user-set value is the `winner` variable in Notebook 03 (intentional — decided by the Task 8 gate). The App `require('../../assets/model/model_int8.onnx')` resolves after Task 20; typed via `as number` with a Task 19 type-check gate. No "similar to Task N" references; code is repeated where shared.
- **Type consistency:** `build_model`/`export_onnx`/`quantize`/`verify` signatures match their callers (compare.py imports `export_onnx`; verify.py uses `build_model`). Python `CLASSES` order equals TS `CLASS_ORDER`; `CONTRACT` stats equal `training/data.py` `MEAN`/`STD`; `verify.py` regenerates the exact fixture the jest parity test consumes; `Prediction.className` matches `getDiseaseInfo` keys; `RootStackParamList` matches all screen `NativeStackScreenProps` uses. `test.csv`/`domain_shift_*` filenames used identically in `make_splits.py`, configs, and `verify.py`.
- **Notebook↔repo contract:** notebooks reference committed modules only (`python -m data.make_manifest`, etc.), so Colab sessions never hand-edit Python.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-08-04-maize-disease-detection.md`.

1. **Subagent-Driven (recommended)** — a fresh subagent executes each task with two-stage review; fast iteration, clean gates.
2. **Inline Execution** — tasks run in this session with checkpoints.

Which approach do you want?
