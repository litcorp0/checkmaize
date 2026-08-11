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
