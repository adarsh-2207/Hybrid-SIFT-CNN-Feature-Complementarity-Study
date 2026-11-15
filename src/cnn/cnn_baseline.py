import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image
from torch.utils.data import Dataset

class Caltech101Dataset(Dataset):

    def __init__(self, root, class_to_idx=None, transform=None):

        self.root = root
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if class_to_idx is None:
            self.class_to_idx = {
                cls_name: idx
                for idx, cls_name in enumerate(self.classes)
            }
        else:
            self.class_to_idx = class_to_idx

        self.samples = []

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

        for class_name in self.classes:

            class_dir = os.path.join(root, class_name)

            for filename in sorted(os.listdir(class_dir)):

                if filename.lower().endswith(valid_extensions):

                    path = os.path.join(class_dir, filename)

                    self.samples.append(
                        (path, self.class_to_idx[class_name])
                    )

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "cnn")

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Train directory: {os.path.join(DATA_DIR, 'train')}")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 60)
print("CNN BASELINE")
print("=" * 60)
print(f"Device: {device}")


# ============================================================
# 4. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 5. DATASETS
# ============================================================

train_dataset = Caltech101Dataset(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

class_to_idx = train_dataset.class_to_idx

val_dataset = Caltech101Dataset(
    os.path.join(DATA_DIR, "val"),
    class_to_idx=class_to_idx,
    transform=eval_transform
)

test_dataset = Caltech101Dataset(
    os.path.join(DATA_DIR, "test"),
    class_to_idx=class_to_idx,
    transform=eval_transform
)


# ============================================================
# 6. VERIFY CLASS MAPPING
# ============================================================

assert train_dataset.class_to_idx == val_dataset.class_to_idx
assert train_dataset.class_to_idx == test_dataset.class_to_idx

num_classes = len(train_dataset.classes)

print(f"Number of classes: {num_classes}")
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 8. MODEL — RESNET-18
# ============================================================

print("\nLoading ResNet-18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Replace ImageNet classifier
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

print("Model: ResNet-18")
print("Pretraining: ImageNet")
print(f"Output classes: {num_classes}")


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    epoch_acc = accuracy_score(
        all_labels,
        all_preds
    )

    epoch_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_acc, epoch_f1


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(loader.dataset)

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return loss, acc, macro_f1, all_labels, all_preds


# ============================================================
# 12. TRAIN
# ============================================================

best_val_f1 = -1

history = []

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_acc, val_f1, _, _ = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "learning_rate": current_lr
    })

    # Save best model based on validation Macro-F1
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            os.path.join(
                RESULTS_DIR,
                "best_model.pth"
            )
        )

        print("  -> Best model saved")


# ============================================================
# 13. SAVE TRAINING HISTORY
# ============================================================

with open(
    os.path.join(RESULTS_DIR, "training_history.json"),
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# 14. LOAD BEST MODEL
# ============================================================

print("\nLoading best validation model...")

model.load_state_dict(
    torch.load(
        os.path.join(
            RESULTS_DIR,
            "best_model.pth"
        ),
        map_location=device
    )
)


# ============================================================
# 15. FINAL VALIDATION
# ============================================================

val_loss, val_acc, val_f1, _, _ = evaluate(
    model,
    val_loader
)

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

print(f"Validation Accuracy : {val_acc:.4f}")
print(f"Validation Macro-F1 : {val_f1:.4f}")


# ============================================================
# 16. FINAL TEST
# ============================================================

test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(
    model,
    test_loader
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_preds,
    target_names=test_dataset.classes,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# 18. SAVE FINAL METRICS
# ============================================================

metrics = {
    "model": "ResNet-18",
    "pretraining": "ImageNet",
    "num_classes": num_classes,
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "test_images": len(test_dataset),
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "validation_accuracy": val_acc,
    "validation_macro_f1": val_f1,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1
}

with open(
    os.path.join(
        RESULTS_DIR,
        "metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


print("\n" + "=" * 60)
print("CNN EXPERIMENT COMPLETE")
print("=" * 60)

print(f"Results saved in: {RESULTS_DIR}")
