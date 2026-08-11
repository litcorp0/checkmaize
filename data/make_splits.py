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
