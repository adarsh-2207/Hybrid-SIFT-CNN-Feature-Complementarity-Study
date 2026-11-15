import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "results", "cnn", "best_model.pth"
)
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "results", "hybrid"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class Caltech101Dataset(torch.utils.data.Dataset):

    def __init__(self, root, class_to_idx, transform=None):
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        )

        for class_name in sorted(class_to_idx.keys()):

            class_dir = os.path.join(root, class_name)

            for filename in sorted(os.listdir(class_dir)):

                if filename.lower().endswith(valid_extensions):

                    self.samples.append(
                        (
                            os.path.join(class_dir, filename),
                            class_to_idx[class_name]
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# Transform
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Build class mapping
# ---------------------------------------------------------

train_root = os.path.join(DATA_ROOT, "train")

classes = sorted([
    d for d in os.listdir(train_root)
    if os.path.isdir(os.path.join(train_root, d))
])

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(classes)
}

print("Classes:", len(classes))


# ---------------------------------------------------------
# Load ResNet-18
# ---------------------------------------------------------

print("Loading ResNet-18...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# ---------------------------------------------------------
# Remove final classifier
# ---------------------------------------------------------

feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

def extract_split(split):

    root = os.path.join(DATA_ROOT, split)

    dataset = Caltech101Dataset(
        root=root,
        class_to_idx=class_to_idx,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    features = []
    labels = []

    print()
    print("=" * 60)
    print(f"Processing {split}")
    print("Images:", len(dataset))
    print("=" * 60)

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(device)

            embeddings = feature_extractor(images)

            embeddings = embeddings.flatten(1)

            features.append(
                embeddings.cpu().numpy()
            )

            labels.append(
                batch_labels.numpy()
            )

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)

    np.save(
        os.path.join(
            OUTPUT_ROOT,
            f"cnn_{split}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_ROOT,
            f"cnn_y_{split}.npy"
        ),
        y
    )

    print(f"{split} feature shape:", X.shape)
    print(f"{split} label shape:", y.shape)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        extract_split(split)

    print()
    print("=" * 60)
    print("CNN feature extraction COMPLETE")
    print("=" * 60)
