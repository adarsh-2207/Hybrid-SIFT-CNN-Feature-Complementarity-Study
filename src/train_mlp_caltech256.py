
import os
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path

# ---------------- CONFIG ----------------

SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
NUM_CLASSES = 256

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cpu")

BASE = Path("results_caltech256")
OUT = BASE / "mlp_updated"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)

# ---------------- MODEL ----------------

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ---------------- DATA ----------------

def load_features(model_type):
    if model_type == "CNN":
        folder = BASE / "hybrid_features"
        files = {
            "train": "cnn_train.npy",
            "val": "cnn_val.npy",
            "test": "cnn_test.npy"
        }

    elif model_type == "SIFT":
        folder = BASE / "sift_bovw_svm"
        files = {
            "train": "X_train.npy",
            "val": "X_val.npy",
            "test": "X_test.npy"
        }

    elif model_type == "HYBRID":
        cnn_folder = BASE / "hybrid_features"
        sift_folder = BASE / "sift_bovw_svm"

        data = {}

        for split in ["train", "val", "test"]:
            cnn = np.load(cnn_folder / f"cnn_{split}.npy")
            sift = np.load(sift_folder / f"X_{split}.npy")

            data[split] = np.concatenate([cnn, sift], axis=1)

        return data

    data = {}

    for split, filename in files.items():
        data[split] = np.load(folder / filename)

    return data


def load_labels():
    labels = {}

    for split in ["train", "val", "test"]:
        labels[split] = np.load(
            BASE / "hybrid_features" / f"cnn_y_{split}.npy"
        ).astype(np.int64)

    return labels


# ---------------- TRAINING ----------------

def train_model(model_type, X, y):

    print(f"\n{'='*55}")
    print(f"Training {model_type} MLP")
    print(f"{'='*55}")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X["train"]).astype(np.float32)
    X_val = scaler.transform(X["val"]).astype(np.float32)
    X_test = scaler.transform(X["test"]).astype(np.float32)

    y_train = y["train"]
    y_val = y["val"]
    y_test = y["test"]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = MLP(X_train.shape[1]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    best_f1 = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_ds)

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

            val_preds = val_logits.argmax(dim=1).cpu().numpy()

        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(
            y_val,
            val_preds,
            average="macro",
            zero_division=0
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate test set only after model selection
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_macro = f1_score(
        y_test, test_preds, average="macro", zero_division=0
    )
    test_weighted = f1_score(
        y_test, test_preds, average="weighted", zero_division=0
    )

    print(f"\n{model_type} TEST RESULTS")
    print(f"Accuracy:   {test_acc:.4f}")
    print(f"Macro-F1:   {test_macro:.4f}")
    print(f"Weighted-F1:{test_weighted:.4f}")
    print(f"Best epoch: {best_epoch}")

    # Save model checkpoint
    torch.save({
        "model_name": model_type,
        "epoch": best_epoch,
        "input_dim": X_train.shape[1],
        "num_classes": NUM_CLASSES,
        "model_state_dict": model.state_dict(),
        "val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro
    }, OUT / f"{model_type.lower()}_mlp_best.pth")

    import joblib
    joblib.dump(scaler, OUT / f"{model_type.lower()}_scaler.joblib")

    pd.DataFrame(history).to_csv(
        OUT / f"{model_type.lower()}_history.csv",
        index=False
    )

    pd.DataFrame({
        "true_label": y_test,
        "predicted_label": test_preds
    }).to_csv(
        OUT / f"{model_type.lower()}_test_predictions.csv",
        index=False
    )

    pd.DataFrame(
        classification_report(
            y_test,
            test_preds,
            output_dict=True,
            zero_division=0
        )
    ).transpose().to_csv(
        OUT / f"{model_type.lower()}_classification_report.csv"
    )

    np.save(
        OUT / f"{model_type.lower()}_confusion_matrix.npy",
        confusion_matrix(y_test, test_preds, labels=np.arange(NUM_CLASSES))
    )

    return {
        "model": model_type,
        "input_dim": X_train.shape[1],
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro,
        "test_weighted_f1": test_weighted
    }


# ---------------- MAIN ----------------

if __name__ == "__main__":

    labels = load_labels()

    # Verify labels and feature dimensions
    for split in ["train", "val", "test"]:
        sift_y = np.load(
            BASE / "sift_bovw_svm" / f"y_{split}.npy"
        )

        assert np.array_equal(labels[split], sift_y)

    results = []

    for model_type in ["CNN", "SIFT", "HYBRID"]:
        features = load_features(model_type)
        result = train_model(model_type, features, labels)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "mlp_comparison.csv", index=False)

    with open(OUT / "mlp_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nALL EXPERIMENTS COMPLETED")
    print(results_df.to_string(index=False))
    print("\nResults saved in:", OUT)
