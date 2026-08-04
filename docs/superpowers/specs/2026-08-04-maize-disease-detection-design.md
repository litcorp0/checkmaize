# Design Spec — Smart Crop Disease Detection System (Maize Leaf Diseases, Ghana)

**Date:** 2026-08-04
**Status:** Approved
**Title:** Smart Crop Disease Detection System Using Machine Learning: A Case Study of Maize Leaf Diseases in Ghana

## 1. Purpose & Scope

Build a working, installable mobile app (Android + iOS) that runs a trained
machine-learning model **fully on-device** to detect and classify maize leaf
diseases found under Ghanaian farming conditions. The app is the primary
deliverable; the Python research pipeline behind it produces rigorous,
reproducible evidence that answers the project's three research questions so a
separate academic writer can author the written study.

The user is a JavaScript/TypeScript developer with no ML or Python background.
The division of labor is therefore explicit:

- **User owns** the Expo/React Native app (TypeScript), the on-device inference
  glue (onnxruntime JS API), and interpreting model-selection numbers with
  assisted explanation.
- **AI assistant (this project) owns** all Python: data pipeline, training,
  benchmarking, export, quantization, verification. The user runs pre-built
  Colab notebooks and relays error output; they never write Python.

### Research Questions Addressed

- **RQ1:** What are the common maize leaf diseases affecting maize production in
  Ghana, and what are their distinguishing visual characteristics?
  → Addressed by the 4-class catalog and the in-app disease knowledge base.
- **RQ2:** How accurately can an ML model detect and classify maize leaf
  diseases from leaf images captured under Ghanaian farming conditions?
  → Addressed by two evaluation axes: within-source Ghana holdout and
    cross-source domain-shift (PlantVillage-trained → Ghana-field test).
- **RQ3:** What ML model architecture is most suitable for an effective and
  computationally efficient system deployable in Ghana?
  → Addressed by a uniform 5-candidate benchmark comparing accuracy vs. ONNX
    size vs. on-device latency.

## 2. Key Decisions (agreed)

| Decision | Choice |
|---|---|
| Primary deliverable | Working installable mobile app running a locally-trained model on-device |
| Platforms | Android + iOS (React Native / Expo) |
| Dataset strategy | Public Ghana (Mendeley) + PlantVillage mix, PLUS small in-country collection via the app |
| Class catalog | 4 classes: Common Rust (CR), Gray Leaf Spot (GLS), Northern Leaf Blight (NLB), Healthy |
| Training compute | Google Colab (free tier) |
| App feature scope | Detect + disease advice + geotagged scan log (all offline) |
| Research write-up | Handled by someone else; this project produces evidence artifacts (data, numbers, figures, reproducibility notes) |
| ML→mobile stack | PyTorch → ONNX → onnxruntime-react-native |

## 3. Architecture & Repository Layout

Five self-contained units with one-way dependency:
`app/` and `benchmarks/` consume `inference/` outputs; `training/` consumes
`data/`; nothing points backward.

```
checkmaize/
├── data/          # dataset ingestion & split configs (no raw data committed)
│   ├── manifests/                 # train/val/test/domain_shift CSVs
│   └── make_splits.py             # stratified splits honoring provenance + leaf_id
├── training/      # PyTorch fine-tuning on Colab
│   ├── configs/                   # yaml per experiment (model, epochs, aug, lr)
│   ├── train.py                   # generic fine-tune driver
│   └── data.py                    # loaders + Ghana-condition augmentation
├── benchmarks/    # RQ3 architecture comparison
│   ├── compare.py                 # uniform protocol over candidate models
│   └── report/                    # accuracy/size/latency tables + plots (academic artifacts)
├── inference/     # PyTorch → ONNX → int8 quantization + parity verification
│   ├── export.py
│   ├── quantize.py
│   └── verify.py                  # numeric parity + quantization delta report
├── app/           # Expo React Native (TypeScript) + onnxruntime-react-native
└── docs/          # design spec, dataset-notes, onnx-contract, pilot-protocol
```

### The Interface Contract (model → app)

The single seam tying ML to mobile is the **ONNX contract**, pinned in
`docs/onnx-contract.md` and mirrored by a single, tested preprocessing module in
the app:

- Fixed input: 224×224 RGB, batch 1.
- Exact preprocessing used in training (resize strategy + normalization stats)
  **must match in JS exactly**.
- Class-index table (CR, GLS, NLB, Healthy) shipped as a `.bin`/JSON alongside
  the model.

## 4. Data Pipeline (`data/`)

**Sources (public; raw data never committed):**
- **Mendeley Ghana maize dataset** — ~3,852 real-field images from Ghana, 4
  classes. The "Ghanaian conditions" anchor. Requires a (free) Mendeley account;
  fallback = Kaggle-hosted mirror or the CD&S Purdue field dataset.
- **PlantVillage maize subset** — ~4,000 images, same 4 classes, controlled
  backgrounds. Volume + domain variety; Kaggle mirrors are fallbacks.
- **Local collection** — captured via the app's Contribute flow, landing in
  `data/raw/local/` as labeled seeds grow.

**Splitting rules (leakage prevention is methodological core):**
- Every image tagged with `(source, leaf_id, class)`; same-leaf images never
  straddle train/test.
- **Within-source:** train on PlantVillage + Ghana slice; test on held-out
  Ghana-only slice.
- **Domain-shift:** train on PlantVillage only; test on the *entire* Mendeley
  Ghana set → strongest RQ2 data point.
- Stratified splits, fixed seed; manifests (CSVs) are the reproducibility
  artifact.

**Outputs:** `data/manifests/{train,val,test,domain_shift}.csv` +
`docs/dataset-notes.md` (per-class counts, provenance, methods).

## 5. Training & RQ3 Benchmark (`training/` + `benchmarks/`)

**Training:** transfer-learning fine-tune of lightweight CNNs; one generic
`train.py` driven by YAML configs; augmentation tuned for field conditions
(rotation, brightness/contrast flicker, blur, color jitter).

**Candidates (`benchmarks/compare.py`):**

| Model | Rationale |
|---|---|
| Custom 2-layer CNN baseline | minimalism floor |
| MobileNetV3-Small | size/accuracy sweet spot |
| MobileNetV3-Large | higher accuracy, still mobile |
| EfficientNet-B0 | best published efficiency-accuracy tradeoff |
| ResNet-18 | heavier reliable baseline |

**Uniform protocol:** same pipeline, augmentation, splits, seed, optim/lr for
all five. Each produces: accuracy, macro-F1, per-class precision/recall,
confusion matrix, ONNX int8 size (MB), on-device latency (ms), params/FLOPs.

**RQ3 answer:** two-axis decision matrix (accuracy ≥ ~95% on Ghana test vs. size
and latency). Expected winner: MobileNetV3-Small or EfficientNet-B0. Tables and
plots → `benchmarks/report/` for the academic writer. Order candidates so weak
ones fail fast; each run fits in one Colab execution.

## 6. Export & Quantization (`inference/`)

1. `export.py` — freeze trained model to ONNX (opset 17, fixed batch 1, 224×224
   RGB); input/output layout written to `docs/onnx-contract.md`.
2. `quantize.py` — int8 quantization via onnxruntime (calibration set = val
   split); shrink ~20 MB → ~3–5 MB.
3. `verify.py` — two gates:
   - **Parity:** PyTorch vs ONNX logits within tolerance.
   - **Quantization delta:** int8 vs fp32 on Ghana test ≤ ~1%, else ship fp32.

**Outputs per candidate** (winner shipped): `model.onnx` (fp32), `model_int8.onnx`,
label index, verification report. Shared with `benchmarks/` — no duplicate work.

## 7. Mobile App (`app/`)

**Stack:** Expo (React Native, TypeScript) + `expo-dev-client` +
`onnxruntime-react-native` + EAS Build (APK + IPA). Managed workflow unsupported
(onnxruntime is a native module → dev build/prebuild required).

**Screens/flows:**
1. **Capture** — camera or gallery import of a single symptomatic leaf (framing
   tips in-shot); auto-resize to 224×224 + exact contract preprocessing.
2. **Result** — top prediction + confidence bar; below threshold → honest
   "unclear, re-capture or consult extension agent."
3. **Disease info** — bundled JSON knowledge base per class (RQ1 distilate):
   visual hallmarks + practical management guidance.
4. **Scan log** — `expo-sqlite`, entries geotagged via `expo-location` (consent
   logged): date, thumbnail, prediction, confidence, GPS.
5. **Contribute** — optional unlabeled-capture save (region metadata) for the
   local-collection loop; exported off-device later, no backend.
6. **About / model-metrics** — shows shipped model's RQ3 numbers; debug screen
   times a single on-device inference for the latency table.

**Anchors:** offline-first (no network in the happy path); preprocessing is a
single tested module; model bundled with the build.

**Known dependency:** onnxruntime-react-native releases pin native versions to
specific Expo SDKs → pin the compatibility matrix (Expo SDK ↔ RN ↔ onnxruntime)
before scaffolding.

## 8. Quality Gates & Testing

**Python (runs on Colab):**
- Data: split-manifest sanity tests (no leaf_id leakage, no source leakage in
  domain-shift, deterministic rebuild).
- Export: parity test + quantization-delta report.
- Benchmark: reference run pinned to a known-good commit for reproducibility.

**App (Jest):**
- **Preprocessing parity test** — money test; JS output must equal Python
  reference tensor (catches the #1 silent on-device bug).
- Inference smoke test — mocked onnxruntime session, asserts
  result/label/confidence mapping.
- Log + geotag unit tests — round-trip persistence; location recorded only with
  consent.

**Field/error handling:**
- Confidence-threshold gate as primary runtime error path.
- Degraded input (blur/dark/duplicate) → "can't tell" result, never a wrong
  confident answer.
- GPS unavailable → `location: null`, scan still saves.

## 9. Deliverables & Field Pilot

**Deliverables:**
1. Android APK + iOS build via EAS; `app/` source; README with install/build
   steps.
2. Research artifacts for the academic writer: dataset notes + provenance, RQ3
   comparison tables/plots, RQ2 Ghana-test results, ONNX contract,
   reproducibility configs — in `benchmarks/report/` and `docs/`.
3. `docs/pilot-protocol.md` — in-country trial: install APK, capture leaves,
   compare app verdicts vs. expert labeling; feed misses into local collection.
4. Local-collection kit: Contribute flow + labeling/merge checklist.

**Execution order (gates):**
1. Pin compatibility matrix; scaffold repo + git init.
2. `data/` — source datasets, manifests + split tests. **Gate:** split sanity
   tests pass.
3. `training/` + `benchmarks/` — Colab notebooks, candidate fine-tunes,
   comparison report. **Gate:** winner chosen by matrix.
4. `inference/` — export, quantize, verify. **Gate:** parity + quantization delta
   pass.
5. `app/` — scaffold, preprocessing + parity test, capture → result → info →
   log → contribute → metrics.
6. EAS builds (APK/IPA), on-device latency, README + pilot protocol.

**Top risks:** Mendeley Ghana dataset access (fallback Kaggle mirror/CD&S); Expo
SDK ↔ onnxruntime version pinning; quantization accuracy drop >1% (fallback ship
fp32).