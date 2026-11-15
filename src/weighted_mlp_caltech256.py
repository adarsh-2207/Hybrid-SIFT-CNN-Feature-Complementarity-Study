
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ---------------- CONFIG ----------------

SEED = 42
BATCH_SIZE = 256
LR = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

BASE = Path("results_caltech256")
OUT = BASE / "weighted_mlp"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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


# ---------------- LOAD FEATURES ----------------

def load_features():
    cnn_dir = BASE / "hybrid_features"
    sift_dir = BASE / "sift_bovw_svm"

    data = {}

    for split in ["train", "val", "test"]:
        data[split] = {
            "cnn": np.load(cnn_dir / f"cnn_{split}.npy"),
            "sift": np.load(sift_dir / f"X_{split}.npy"),
            "y": np.load(cnn_dir / f"cnn_y_{split}.npy").astype(np.int64)
        }

        sift_y = np.load(sift_dir / f"y_{split}.npy")
        assert np.array_equal(data[split]["y"], sift_y)

    return data


# ---------------- TRAIN ONE BETA ----------------

def run_experiment(data, beta):

    print(f"\n{'='*55}")
    print(f"Training weighted Hybrid MLP | beta={beta}")
    print(f"{'='*55}")

    # Fit separate scalers on training data only
    cnn_scaler = StandardScaler()
    sift_scaler = StandardScaler()

    cnn_train = cnn_scaler.fit_transform(
        data["train"]["cnn"]
    ).astype(np.float32)

    sift_train = sift_scaler.fit_transform(
        data["train"]["sift"]
    ).astype(np.float32)

    cnn_val = cnn_scaler.transform(
        data["val"]["cnn"]
    ).astype(np.float32)

    sift_val = sift_scaler.transform(
        data["val"]["sift"]
    ).astype(np.float32)

    cnn_test = cnn_scaler.transform(
        data["test"]["cnn"]
    ).astype(np.float32)

    sift_test = sift_scaler.transform(
        data["test"]["sift"]
    ).astype(np.float32)

    # Weighted concatenation
    X_train = np.concatenate(
        [cnn_train, beta * sift_train], axis=1
    )

    X_val = np.concatenate(
        [cnn_val, beta * sift_val], axis=1
    )

    X_test = np.concatenate(
        [cnn_test, beta * sift_test], axis=1
    )

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    generator = torch.Generator().manual_seed(SEED)

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = MLP(X_train.shape[1]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    Xv = torch.from_numpy(X_val).to(DEVICE)
    yv = torch.from_numpy(y_val).to(DEVICE)

    best_f1 = -1
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_loss = 0

        for xb, yb in loader:
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
            val_logits = model(Xv)
            val_loss = criterion(val_logits, yv).item()
            val_pred = val_logits.argmax(1).cpu().numpy()

        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(
            y_val, val_pred,
            average="macro", zero_division=0
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
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Test only after selecting the best epoch
    Xt = torch.from_numpy(X_test).to(DEVICE)

    with torch.no_grad():
        test_logits = model(Xt)
        test_pred = test_logits.argmax(1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(
        y_test, test_pred,
        average="macro", zero_division=0
    )
    test_weighted = f1_score(
        y_test, test_pred,
        average="weighted", zero_division=0
    )

    print(f"\nBeta {beta} TEST RESULTS")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    print(f"Weighted-F1: {test_weighted:.4f}")

    tag = str(beta).replace(".", "_")

    torch.save({
        "beta": beta,
        "epoch": best_epoch,
        "input_dim": X_train.shape[1],
        "model_state_dict": model.state_dict(),
        "val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1
    }, OUT / f"weighted_mlp_beta_{tag}.pth")

    joblib.dump(
        cnn_scaler, OUT / f"cnn_scaler_beta_{tag}.joblib"
    )
    joblib.dump(
        sift_scaler, OUT / f"sift_scaler_beta_{tag}.joblib"
    )

    pd.DataFrame(history).to_csv(
        OUT / f"history_beta_{tag}.csv", index=False
    )

    pd.DataFrame({
        "true_label": y_test,
        "predicted_label": test_pred
    }).to_csv(
        OUT / f"predictions_beta_{tag}.csv", index=False
    )

    return {
        "beta": beta,
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_weighted_f1": test_weighted
    }


# ---------------- MAIN ----------------

if __name__ == "__main__":

    data = load_features()

    print("\nFeature shapes:")
    for split in ["train", "val", "test"]:
        print(
            split,
            data[split]["cnn"].shape,
            data[split]["sift"].shape
        )

    results = []

    for beta in BETAS:
        results.append(run_experiment(data, beta))

    df = pd.DataFrame(results)

    # Select beta using validation Macro-F1 only
    best_idx = df["val_macro_f1"].idxmax()
    best = df.loc[best_idx]

    df.to_csv(OUT / "weighted_mlp_comparison.csv", index=False)

    with open(OUT / "best_beta.json", "w") as f:
        json.dump(best.to_dict(), f, indent=4)

    print("\nFINAL COMPARISON")
    print(df.to_string(index=False))

    print("\nBEST BETA SELECTED ON VALIDATION:")
    print(best.to_string())

    print("\nResults saved to:", OUT)
