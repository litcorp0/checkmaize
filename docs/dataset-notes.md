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
