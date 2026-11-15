import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-3
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

##RESULTS_DIR = "results/cnn_mobilenet"
##RESULTS_DIR = "results/cnn_mobilenet_wd"
##RESULTS_DIR = "results/cnn_mobilenet_augmented"
RESULTS_DIR = "results/cnn_mobilenet_augmented_wd"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class Caltech101Dataset(Dataset):

    def __init__(self, root, class_to_idx, transform=None):
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []

        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(root, class_name)

            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".gif")
                ):
                    self.samples.append((path, class_idx))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_classes(root):
    classes = sorted(
        [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
    )

    return {name: idx for idx, name in enumerate(classes)}


class_to_idx = get_classes(TRAIN_DIR)

print("Number of classes:", len(class_to_idx))


# ============================================================
# TRANSFORMS
# Same protocol as ResNet-18 baseline
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.20,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value=0
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = Caltech101Dataset(
    TRAIN_DIR,
    class_to_idx,
    train_transform
)

val_dataset = Caltech101Dataset(
    VAL_DIR,
    class_to_idx,
    eval_transform
)

test_dataset = Caltech101Dataset(
    TEST_DIR,
    class_to_idx,
    eval_transform
)

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = mobilenet_v3_small(weights=weights)

num_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    num_features,
    len(class_to_idx)
)

model = model.to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

##criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    loss = total_loss / len(loader.dataset)

    return loss, accuracy, macro_f1


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1
best_epoch = -1

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    train_preds = []
    train_labels = []

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        am = np.random.beta(0.2, 0.2)
        indices = torch.randperm(images.size(0), device=images.device)
        lam = np.random.beta(0.2, 0.2)
        indices = torch.randperm(images.size(0), device=images.device)
        
        mixed_images = (lam * images + (1 - lam) * images[indices])
        labels_a = labels
        labels_b = labels[indices]

        outputs = model(mixed_images)

        loss = (
            lam * criterion(outputs, labels_a)
            + (1 - lam) * criterion(outputs, labels_b)
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)

    train_acc = accuracy_score(
        train_labels,
        train_preds
    )

    train_f1 = f1_score(
        train_labels,
        train_preds,
        average="macro",
        zero_division=0
    )

    val_loss, val_acc, val_f1 = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch:02d}: "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Train F1 {train_f1:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"Val F1 {val_f1:.4f} | "
        f"LR {current_lr:.2e}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "learning_rate": current_lr
    })

    # --------------------------------------------------------
    # BEST CHECKPOINT = validation Macro-F1
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "val_macro_f1": val_f1
            },
            os.path.join(
                RESULTS_DIR,
                "best_model.pth"
            )
        )

        print("  --> Saved BEST model")


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    os.path.join(RESULTS_DIR, "training_history.json"),
    "w"
) as f:

    json.dump(history, f, indent=2)


# ============================================================
# FINAL TEST
# Load BEST validation checkpoint
# ============================================================

checkpoint = torch.load(
    os.path.join(
        RESULTS_DIR,
        "best_model.pth"
    ),
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_acc, test_f1 = evaluate(
    model,
    test_loader
)


# ============================================================
# RESULTS
# ============================================================

results = {
    "model": "MobileNetV3-Small",
    "dataset": "Caltech-101",
    "seed": SEED,
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_val_f1,
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY
}

with open(
    os.path.join(RESULTS_DIR, "metrics.json"),
    "w"
) as f:

    json.dump(results, f, indent=2)


print("\n" + "=" * 70)
print("MOBILENETV3-SMALL FINAL RESULTS")
print("=" * 70)

print(f"Best Epoch       : {best_epoch}")
print(f"Best Val Macro-F1: {best_val_f1 * 100:.2f}%")
print(f"Test Accuracy    : {test_acc * 100:.2f}%")
print(f"Test Macro-F1    : {test_f1 * 100:.2f}%")

print("\nSaved:")
print(f"  {RESULTS_DIR}/best_model.pth")
print(f"  {RESULTS_DIR}/training_history.json")
print(f"  {RESULTS_DIR}/metrics.json")
