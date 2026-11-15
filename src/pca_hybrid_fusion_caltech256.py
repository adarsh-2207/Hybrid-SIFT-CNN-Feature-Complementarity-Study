
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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42
NUM_CLASSES = 256

PCA_COMPONENTS = [32, 64, 128, 256]

BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10

BASE = Path("results_caltech256")

CNN_DIR = BASE / "hybrid_features"
SIFT_DIR = BASE / "sift_bovw_svm"

OUT = BASE / "pca_hybrid_fusion"
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
# MLP ARCHITECTURE
# Same architecture as previous MLP experiments
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

    data = {}

    for split in ["train", "val", "test"]:

        cnn = np.load(
            CNN_DIR / f"cnn_{split}.npy"
        ).astype(np.float32)

        sift = np.load(
            SIFT_DIR / f"X_{split}.npy"
        ).astype(np.float32)

        y = np.load(
            CNN_DIR / f"cnn_y_{split}.npy"
        ).astype(np.int64)

        sift_y = np.load(
            SIFT_DIR / f"y_{split}.npy"
        ).astype(np.int64)

        assert np.array_equal(y, sift_y), (
            f"Label mismatch in {split}"
        )

        assert len(cnn) == len(sift) == len(y)

        data[split] = {
            "cnn": cnn,
            "sift": sift,
            "y": y
        }

        print(
            f"{split}: CNN={cnn.shape}, "
            f"SIFT={sift.shape}, y={y.shape}"
        )

    assert set(np.unique(data["train"]["y"])) == set(
        range(NUM_CLASSES)
    )

    return data


# =========================================================
# MLP TRAINING FUNCTION
# =========================================================

def train_mlp(X_train, y_train, X_val, y_val, name):

    print("\n" + "=" * 65)
    print("TRAINING:", name)
    print("=" * 65)

    set_seed(SEED)

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64))
    )

    generator = torch.Generator().manual_seed(SEED)

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = MLP(
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    Xv = torch.from_numpy(
        X_val.astype(np.float32)
    ).to(DEVICE)

    yv = torch.from_numpy(
        y_val.astype(np.int64)
    ).to(DEVICE)

    best_f1 = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0

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

            val_loss = criterion(
                val_logits, yv
            ).item()

            val_pred = val_logits.argmax(dim=1).cpu().numpy()

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

    model.load_state_dict(best_state)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1
    }, OUT / f"{name}_best.pth")

    pd.DataFrame(history).to_csv(
        OUT / f"{name}_history.csv",
        index=False
    )

    print(
        f"\n{name} best epoch: {best_epoch}"
    )

    print(
        f"{name} best validation Macro-F1: {best_f1:.4f}"
    )

    return model, {
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1
    }


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(model, X):

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(0, len(X), 512):

            xb = torch.from_numpy(
                X[start:start + 512].astype(np.float32)
            ).to(DEVICE)

            logits = model(xb)

            predictions.extend(
                logits.argmax(dim=1).cpu().numpy()
            )

    return np.array(predictions)


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    data = load_data()

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    # -----------------------------------------------------
    # CNN STANDARDIZATION
    # -----------------------------------------------------

    print("\nStandardizing CNN features...")

    cnn_scaler = StandardScaler()

    cnn_train = cnn_scaler.fit_transform(
        data["train"]["cnn"]
    ).astype(np.float32)

    cnn_val = cnn_scaler.transform(
        data["val"]["cnn"]
    ).astype(np.float32)

    cnn_test = cnn_scaler.transform(
        data["test"]["cnn"]
    ).astype(np.float32)

    joblib.dump(
        cnn_scaler,
        OUT / "cnn_scaler.joblib"
    )

    # -----------------------------------------------------
    # CNN-ONLY CONTROL
    # -----------------------------------------------------

    cnn_model, cnn_info = train_mlp(
        cnn_train,
        y_train,
        cnn_val,
        y_val,
        "cnn_only"
    )

    cnn_val_pred = predict(cnn_model, cnn_val)
    cnn_test_pred = predict(cnn_model, cnn_test)

    # -----------------------------------------------------
    # SIFT STANDARDIZATION
    # -----------------------------------------------------

    print("\nStandardizing SIFT features...")

    sift_scaler = StandardScaler()

    sift_train = sift_scaler.fit_transform(
        data["train"]["sift"]
    ).astype(np.float32)

    sift_val = sift_scaler.transform(
        data["val"]["sift"]
    ).astype(np.float32)

    sift_test = sift_scaler.transform(
        data["test"]["sift"]
    ).astype(np.float32)

    joblib.dump(
        sift_scaler,
        OUT / "sift_scaler.joblib"
    )

    # -----------------------------------------------------
    # PCA EXPERIMENTS
    # -----------------------------------------------------

    experiment_results = []

    trained_models = {}

    for components in PCA_COMPONENTS:

        print("\n" + "#" * 65)
        print(f"PCA COMPONENTS: {components}")
        print("#" * 65)

        pca = PCA(
            n_components=components,
            random_state=SEED
        )

        # Fit PCA only on training SIFT features
        sift_train_pca = pca.fit_transform(
            sift_train
        )

        sift_val_pca = pca.transform(sift_val)
        sift_test_pca = pca.transform(sift_test)

        explained_variance = float(
            pca.explained_variance_ratio_.sum()
        )

        # Concatenate CNN and reduced SIFT features
        X_train = np.concatenate(
            [cnn_train, sift_train_pca],
            axis=1
        ).astype(np.float32)

        X_val = np.concatenate(
            [cnn_val, sift_val_pca],
            axis=1
        ).astype(np.float32)

        X_test = np.concatenate(
            [cnn_test, sift_test_pca],
            axis=1
        ).astype(np.float32)

        print("Hybrid feature shape:", X_train.shape)

        print(
            "SIFT explained variance:",
            round(explained_variance, 5)
        )

        joblib.dump(
            pca,
            OUT / f"sift_pca_{components}.joblib"
        )

        # Train MLP
        name = f"pca_{components}"

        model, info = train_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            name
        )

        val_pred = predict(model, X_val)

        val_acc = accuracy_score(y_val, val_pred)

        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro",
            zero_division=0
        )

        experiment_results.append({
            "experiment": name,
            "sift_components": components,
            "total_dimensions": X_train.shape[1],
            "explained_variance": explained_variance,
            "best_epoch": info["best_epoch"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        trained_models[components] = {
            "model": model,
            "test_features": X_test
        }

    # -----------------------------------------------------
    # VALIDATION COMPARISON
    # -----------------------------------------------------

    results_df = pd.DataFrame(experiment_results)

    results_df.to_csv(
        OUT / "pca_validation_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("VALIDATION COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    # Select best PCA configuration using validation Macro-F1
    results_df = results_df.sort_values(
        ["val_macro_f1", "val_accuracy", "sift_components"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    best = results_df.iloc[0]

    best_components = int(
        best["sift_components"]
    )

    print("\nSELECTED PCA COMPONENTS:", best_components)

    # -----------------------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------------------

    selected = trained_models[best_components]

    hybrid_test_pred = predict(
        selected["model"],
        selected["test_features"]
    )

    results = []

    for name, pred in [
        ("CNN_only", cnn_test_pred),
        (f"PCA_Hybrid_{best_components}", hybrid_test_pred)
    ]:

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0
            ),
            "weighted_f1": f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )
        })

    final_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(final_df.to_string(index=False))

    final_df.to_csv(
        OUT / "final_test_results.csv",
        index=False
    )

    pd.DataFrame({
        "true_label": y_test,
        "cnn_prediction": cnn_test_pred,
        "hybrid_prediction": hybrid_test_pred
    }).to_csv(
        OUT / "test_predictions.csv",
        index=False
    )

    with open(OUT / "final_results.json", "w") as f:

        json.dump({
            "selected_pca_components": best_components,
            "selected_total_dimensions": int(
                best["total_dimensions"]
            ),
            "selected_explained_variance": float(
                best["explained_variance"]
            ),
            "selection_metric": "validation_macro_f1",
            "cnn_control": cnn_info,
            "test_results": results
        }, f, indent=4)

    print("\nExperiment completed successfully.")
    print("Results saved in:", OUT)


if __name__ == "__main__":
    main()

