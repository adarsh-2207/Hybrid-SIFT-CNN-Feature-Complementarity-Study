
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
NUM_CLASSES = 256

ALPHAS = np.round(np.arange(0.0, 1.001, 0.05), 2)

BASE = Path("results_caltech256")
OUT = BASE / "late_fusion_mlp"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# MLP ARCHITECTURE — SAME AS PREVIOUS EXPERIMENT
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    cnn_dir = BASE / "hybrid_features"
    sift_dir = BASE / "sift_bovw_svm"

    data = {}

    for split in ["train", "val", "test"]:

        cnn = np.load(cnn_dir / f"cnn_{split}.npy")
        sift = np.load(sift_dir / f"X_{split}.npy")

        y = np.load(
            cnn_dir / f"cnn_y_{split}.npy"
        ).astype(np.int64)

        sift_y = np.load(
            sift_dir / f"y_{split}.npy"
        ).astype(np.int64)

        assert np.array_equal(y, sift_y), (
            f"Label mismatch in {split}"
        )

        assert cnn.shape[0] == len(y)
        assert sift.shape[0] == len(y)

        data[split] = {
            "cnn": cnn.astype(np.float32),
            "sift": sift.astype(np.float32),
            "y": y
        }

        print(
            f"{split}: CNN={cnn.shape}, "
            f"SIFT={sift.shape}, labels={y.shape}"
        )

    assert set(np.unique(data["train"]["y"])) == set(range(NUM_CLASSES)), (
        "Training labels must contain class IDs 0 to 255"
    )

    return data


# =========================================================
# TRAIN ONE BRANCH
# =========================================================

def train_branch(branch, data):

    print("\n" + "=" * 60)
    print(f"TRAINING {branch} MLP")
    print("=" * 60)

    set_seed(SEED)

    # Separate scaler for each feature branch
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        data["train"][branch]
    ).astype(np.float32)

    X_val = scaler.transform(
        data["val"][branch]
    ).astype(np.float32)

    X_test = scaler.transform(
        data["test"][branch]
    ).astype(np.float32)

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]

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

    model = MLP(X_train.shape[1], NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    Xv = torch.from_numpy(X_val).to(DEVICE)
    yv = torch.from_numpy(y_val).to(DEVICE)

    best_f1 = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = []

    # ---------------- TRAINING LOOP ----------------

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

            val_probs = torch.softmax(
                val_logits, dim=1
            ).cpu().numpy()

            val_pred = val_probs.argmax(axis=1)

        val_acc = accuracy_score(y_val, val_pred)

        val_f1 = f1_score(
            y_val,
            val_pred,
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

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Obtain probabilities from the selected model
    def predict_probabilities(X):

        probabilities = []

        with torch.no_grad():

            for start in range(0, len(X), 512):

                batch = torch.from_numpy(
                    X[start:start + 512]
                ).to(DEVICE)

                logits = model(batch)

                probs = torch.softmax(
                    logits, dim=1
                )

                probabilities.append(
                    probs.cpu().numpy()
                )

        return np.concatenate(probabilities, axis=0)

    val_probs = predict_probabilities(X_val)
    test_probs = predict_probabilities(X_test)

    # Save branch checkpoint and scaler
    torch.save({
        "branch": branch,
        "epoch": best_epoch,
        "input_dim": X_train.shape[1],
        "num_classes": NUM_CLASSES,
        "model_state_dict": model.state_dict(),
        "val_macro_f1": best_f1
    }, OUT / f"{branch}_mlp_best.pth")

    joblib.dump(
        scaler,
        OUT / f"{branch}_scaler.joblib"
    )

    pd.DataFrame(history).to_csv(
        OUT / f"{branch}_history.csv",
        index=False
    )

    np.save(
        OUT / f"{branch}_val_probabilities.npy",
        val_probs
    )

    np.save(
        OUT / f"{branch}_test_probabilities.npy",
        test_probs
    )

    print(f"\n{branch} best epoch: {best_epoch}")
    print(f"{branch} validation Macro-F1: {best_f1:.4f}")

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1
    }


# =========================================================
# LATE FUSION
# =========================================================

def main():

    data = load_data()

    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    # Train each branch only once
    cnn = train_branch("cnn", data)
    sift = train_branch("sift", data)

    # ---------------- VALIDATION ALPHA SEARCH ----------------

    print("\n" + "=" * 60)
    print("VALIDATION-BASED ALPHA SELECTION")
    print("=" * 60)

    alpha_results = []

    for alpha in ALPHAS:

        fused_val = (
            alpha * cnn["val_probs"]
            + (1 - alpha) * sift["val_probs"]
        )

        pred = fused_val.argmax(axis=1)

        acc = accuracy_score(y_val, pred)

        macro = f1_score(
            y_val,
            pred,
            average="macro",
            zero_division=0
        )

        weighted = f1_score(
            y_val,
            pred,
            average="weighted",
            zero_division=0
        )

        alpha_results.append({
            "alpha_cnn": float(alpha),
            "alpha_sift": float(1 - alpha),
            "val_accuracy": acc,
            "val_macro_f1": macro,
            "val_weighted_f1": weighted
        })

        print(
            f"Alpha CNN={alpha:.2f} | "
            f"Alpha SIFT={1-alpha:.2f} | "
            f"Val Acc={acc:.4f} | "
            f"Val Macro-F1={macro:.4f}"
        )

    alpha_df = pd.DataFrame(alpha_results)

    alpha_df.to_csv(
        OUT / "validation_alpha_results.csv",
        index=False
    )

    # Select using validation Macro-F1.
    # Tie-break using validation accuracy.
    alpha_df = alpha_df.sort_values(
        ["val_macro_f1", "val_accuracy", "alpha_cnn"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best = alpha_df.iloc[0]

    best_alpha = float(best["alpha_cnn"])

    print("\nSELECTED ALPHA:", best_alpha)

    # ---------------- TEST EVALUATION ----------------

    # Test is used only after alpha selection
    fused_test = (
        best_alpha * cnn["test_probs"]
        + (1 - best_alpha) * sift["test_probs"]
    )

    final_pred = fused_test.argmax(axis=1)

    # Baseline predictions
    cnn_pred = cnn["test_probs"].argmax(axis=1)
    sift_pred = sift["test_probs"].argmax(axis=1)

    results = []

    for name, pred in [
        ("CNN_only", cnn_pred),
        ("SIFT_only", sift_pred),
        ("Late_fusion", final_pred)
    ]:

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test, pred,
                average="macro",
                zero_division=0
            ),
            "weighted_f1": f1_score(
                y_test, pred,
                average="weighted",
                zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    # Save selected alpha and final results
    with open(OUT / "final_results.json", "w") as f:

        json.dump({
            "selected_alpha_cnn": best_alpha,
            "selected_alpha_sift": 1 - best_alpha,
            "selection_metric": "validation_macro_f1",
            "cnn_best_epoch": cnn["best_epoch"],
            "sift_best_epoch": sift["best_epoch"],
            "results": results
        }, f, indent=4)

    results_df.to_csv(
        OUT / "final_results.csv",
        index=False
    )

    # Save final predictions and probabilities
    pd.DataFrame({
        "true_label": y_test,
        "predicted_label": final_pred
    }).to_csv(
        OUT / "late_fusion_test_predictions.csv",
        index=False
    )

    np.save(
        OUT / "late_fusion_test_probabilities.npy",
        fused_test
    )

    np.save(
        OUT / "late_fusion_test_predictions.npy",
        final_pred
    )

    # Classification report
    report = classification_report(
        y_test,
        final_pred,
        labels=np.arange(NUM_CLASSES),
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUT / "late_fusion_classification_report.csv"
    )

    np.save(
        OUT / "late_fusion_confusion_matrix.npy",
        confusion_matrix(
            y_test,
            final_pred,
            labels=np.arange(NUM_CLASSES)
        )
    )

    print("\nExperiment completed successfully.")
    print("Results saved in:", OUT)


if __name__ == "__main__":
    main()
