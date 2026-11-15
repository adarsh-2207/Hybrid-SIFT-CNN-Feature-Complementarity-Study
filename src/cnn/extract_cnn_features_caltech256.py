"""Extract pooled MobileNetV3-Small embeddings for the fixed Caltech-256 split.

This script is intentionally separate from the existing Caltech-101 feature extractor.
It uses the exact sorted class/file ordering used by the Caltech-256 SIFT-BoVW script,
checks class mapping and sample counts, and saves image paths for alignment auditing.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data_caltech256"
MODEL_PATH = PROJECT_ROOT / "results_caltech256" / "mobilenetv3_small" / "best_model.pth"
SIFT_ROOT = PROJECT_ROOT / "results_caltech256" / "sift_bovw_svm"
OUTPUT_ROOT = PROJECT_ROOT / "results_caltech256" / "hybrid_features"

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"
}
BATCH_SIZE = 32


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class OrderedImageDataset(Dataset):
    def __init__(self, split_root: Path, class_to_idx: dict):
        self.samples = []
        for class_name in sorted(class_to_idx):
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.iterdir(), key=lambda p: p.name):
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((path.resolve(), class_to_idx[class_name]))
        self.transform = TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label


def load_class_mapping():
    train_root = DATA_ROOT / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {train_root}")
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    class_to_idx = {name: index for index, name in enumerate(classes)}

    sift_mapping_path = SIFT_ROOT / "class_to_idx.json"
    if not sift_mapping_path.is_file():
        raise FileNotFoundError(
            f"Cannot find SIFT class mapping: {sift_mapping_path}. Run the Caltech-256 SIFT script first."
        )
    with sift_mapping_path.open("r", encoding="utf-8") as file:
        sift_mapping = json.load(file)
    sift_mapping = {str(k): int(v) for k, v in sift_mapping.items()}
    if class_to_idx != sift_mapping:
        raise ValueError("CNN and SIFT class mappings differ. Stop: features must use identical labels.")
    return class_to_idx


def load_model(num_classes: int):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MobileNet checkpoint not found: {MODEL_PATH}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    # Support either a raw state_dict or a common checkpoint dictionary.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and checkpoint and all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


def extract_split(model, class_to_idx: dict, split: str):
    split_root = DATA_ROOT / split
    dataset = OrderedImageDataset(split_root, class_to_idx)
    if not dataset.samples:
        raise RuntimeError(f"No images found in {split_root}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_features, all_labels = [], []
    print(f"\nExtracting {split}: {len(dataset):,} images")

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(DEVICE)
            # MobileNetV3-Small pooled backbone embedding: expected shape [batch, 576].
            embeddings = model.avgpool(model.features(images)).flatten(1)
            all_features.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(labels.numpy().astype(np.int64, copy=False))
            if batch_index % 100 == 0:
                print(f"  processed {min(batch_index * BATCH_SIZE, len(dataset)):,}/{len(dataset):,}")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    paths = [str(path) for path, _ in dataset.samples]

    if X.shape != (len(dataset), 576):
        raise RuntimeError(f"Unexpected embedding shape for {split}: {X.shape}; expected (N, 576)")
    expected_labels = np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    if not np.array_equal(y, expected_labels):
        raise RuntimeError(f"Label order changed during extraction for {split}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_ROOT / f"cnn_{split}.npy", X)
    np.save(OUTPUT_ROOT / f"cnn_y_{split}.npy", y)
    with (OUTPUT_ROOT / f"cnn_paths_{split}.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(paths) + "\n")

    # Verify that saved SIFT arrays have matching row counts and labels.
    sift_X_path = SIFT_ROOT / f"X_{split}.npy"
    sift_y_path = SIFT_ROOT / f"y_{split}.npy"
    if sift_X_path.is_file() and sift_y_path.is_file():
        sift_X = np.load(sift_X_path, mmap_mode="r")
        sift_y = np.load(sift_y_path)
        if len(sift_X) != len(X) or not np.array_equal(sift_y, y):
            raise ValueError(
                f"{split}: SIFT/CNN counts or label order differ. Do not concatenate these features."
            )
        print(f"  SIFT alignment check: row count and labels MATCH ({len(y):,} rows)")
    else:
        raise FileNotFoundError(f"Missing SIFT feature/label files for {split} in {SIFT_ROOT}")

    print(f"  CNN feature shape: {X.shape}")
    print(f"  Label shape:       {y.shape}")
    print(f"  Saved paths:       cnn_paths_{split}.txt")


def main():
    print("Device:", DEVICE)
    print("Dataset:", DATA_ROOT)
    print("Checkpoint:", MODEL_PATH)
    print("Output:", OUTPUT_ROOT)

    class_to_idx = load_class_mapping()
    print("Classes:", len(class_to_idx))
    model = load_model(len(class_to_idx))

    for split in ("train", "val", "test"):
        extract_split(model, class_to_idx, split)

    print("\nCNN feature extraction complete.")
    print("Note: matching labels/counts are checked; path-level SIFT alignment should also be verified before fusion.")


if __name__ == "__main__":
    main()
