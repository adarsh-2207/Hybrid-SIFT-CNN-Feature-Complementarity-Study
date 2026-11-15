#!/usr/bin/env python3
"""Train matched CNN-only and CNN+SIFT classifiers for Caltech-256.

Uses frozen feature matrices and the existing fixed train/val/test split.
Preprocessing is fit on training data only. C=1.0 is fixed in advance;
validation metrics are reported for comparison, and test metrics are final
reporting only (not used for model selection).
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "results_caltech256" / "hybrid_features"
SIFT_ROOT = PROJECT_ROOT / "results_caltech256" / "sift_bovw_svm"
OUTPUT_ROOT = PROJECT_ROOT / "results_caltech256" / "hybrid_classifier"
RANDOM_SEED = 42
LOGREG_C = 1.0
MAX_ITER = 1000

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_split(split):
    X_cnn = np.load(FEATURE_ROOT / f"cnn_{split}.npy")
    y_cnn = np.load(FEATURE_ROOT / f"cnn_y_{split}.npy")
    X_sift = np.load(SIFT_ROOT / f"X_{split}.npy")
    y_sift = np.load(SIFT_ROOT / f"y_{split}.npy")

    if len(X_cnn) != len(X_sift):
        raise ValueError(f"{split}: CNN/SIFT row counts differ.")
    if not np.array_equal(y_cnn, y_sift):
        raise ValueError(f"{split}: CNN/SIFT labels differ.")
    if len(X_cnn) != len(y_cnn) or len(X_sift) != len(y_sift):
        raise ValueError(f"{split}: feature/label row counts differ.")

    return X_cnn, X_sift, y_cnn


def make_classifier():
    # Scaler is fitted only on the training matrix by the Pipeline.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=LOGREG_C,
            max_iter=MAX_ITER,
            solver="lbfgs",
            random_state=RANDOM_SEED,
            verbose=0,
        )),
    ])


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
    }


def main():
    print("=" * 72)
    print("CALTECH-256: MATCHED CNN-ONLY vs CNN+SIFT CLASSIFIERS")
    print("=" * 72)

    X_cnn_train, X_sift_train, y_train = load_split("train")
    X_cnn_val, X_sift_val, y_val = load_split("val")
    X_cnn_test, X_sift_test, y_test = load_split("test")

    X_hybrid_train = np.concatenate([X_cnn_train, X_sift_train], axis=1)
    X_hybrid_val = np.concatenate([X_cnn_val, X_sift_val], axis=1)
    X_hybrid_test = np.concatenate([X_cnn_test, X_sift_test], axis=1)

    X_sift_train_saved = X_sift_train
    X_sift_val_saved = X_sift_val
    X_sift_test_saved = X_sift_test

    datasets = {
        "cnn_only": (X_cnn_train, X_cnn_val, X_cnn_test),
        "sift_logreg": (
            X_sift_train_saved, X_sift_val_saved, X_sift_test_saved
        ),
        "hybrid_cnn_sift": (
            X_hybrid_train, X_hybrid_val, X_hybrid_test
        ),
    }

    print(f"Train: {len(y_train):,} | Validation: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"CNN dimensions: {X_cnn_train.shape[1]}")
    print(f"SIFT dimensions: {X_sift_train.shape[1]}")
    print(f"Hybrid dimensions: {X_hybrid_train.shape[1]}")
    print(f"LogisticRegression C={LOGREG_C}, max_iter={MAX_ITER}")
    print("Feature scaling: StandardScaler, fitted on training data only.")

    all_metrics = {
        "dataset": "Caltech-256",
        "random_seed": RANDOM_SEED,
        "classifier": "StandardScaler + LogisticRegression(lbfgs)",
        "C": LOGREG_C,
        "max_iter": MAX_ITER,
        "feature_dimensions": {
            "cnn_only": int(X_cnn_train.shape[1]),
            "sift_logreg": int(X_sift_train.shape[1]),
            "hybrid_cnn_sift": int(X_hybrid_train.shape[1]),
        },
        "splits": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "models": {},
        "note": (
            "Validation metrics are for comparison. Test metrics are reported "
            "for the fixed configurations and must not be used for tuning."
        ),
    }

    fitted_models = {}
    for name, (X_train, X_val, X_test) in datasets.items():
        print(f"\n{'-' * 72}\nTraining: {name}\n{'-' * 72}")
        model = make_classifier()
        model.fit(X_train, y_train)
        fitted_models[name] = model

        val_pred = model.predict(X_val)
        val_metrics = score(y_val, val_pred)
        print(
            f"Validation accuracy: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Macro-F1: {val_metrics['macro_f1'] * 100:.2f}%"
        )

        # Test is evaluated once for each pre-specified model configuration.
        test_pred = model.predict(X_test)
        test_metrics = score(y_test, test_pred)
        print(
            f"Test accuracy:       {test_metrics['accuracy'] * 100:.2f}% | "
            f"Macro-F1: {test_metrics['macro_f1'] * 100:.2f}%"
        )

        all_metrics["models"][name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        np.save(OUTPUT_ROOT / f"{name}_test_predictions.npy", test_pred)
        cm = confusion_matrix(y_test, test_pred, labels=np.arange(256))
        np.save(OUTPUT_ROOT / f"{name}_confusion_matrix.npy", cm)
        np.savetxt(
            OUTPUT_ROOT / f"{name}_confusion_matrix.csv",
            cm, delimiter=",", fmt="%d"
        )

        with open(OUTPUT_ROOT / f"{name}_test_classification_report.json", "w") as f:
            json.dump(
                classification_report(
                    y_test, test_pred, labels=np.arange(256),
                    output_dict=True, zero_division=0
                ),
                f, indent=2
            )

    # Save a compact table for convenient reporting.
    with open(OUTPUT_ROOT / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "feature_dimensions",
            "validation_accuracy", "validation_macro_f1",
            "test_accuracy", "test_macro_f1",
        ])
        for name, result in all_metrics["models"].items():
            writer.writerow([
                name,
                all_metrics["feature_dimensions"][name],
                result["validation"]["accuracy"],
                result["validation"]["macro_f1"],
                result["test"]["accuracy"],
                result["test"]["macro_f1"],
            ])

    with open(OUTPUT_ROOT / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved comparison and diagnostics to:\n{OUTPUT_ROOT}")
    print("\nReminder: compare CNN-only and hybrid as the matched primary comparison.")
    print("SIFT + RBF-SVM from the existing baseline is a separate traditional baseline.")


if __name__ == "__main__":
    main()
