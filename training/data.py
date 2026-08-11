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
