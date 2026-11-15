
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "results" / "hybrid"
OUT_DIR = ROOT / "results" / "weighted_fusion_caltech101"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load saved features
splits = ["train", "val", "test"]

cnn = {
    s: np.load(FEATURE_DIR / f"cnn_{s}.npy")
    for s in splits
}

sift = {
    s: np.load(FEATURE_DIR / f"sift_{s}.npy")
    for s in splits
}

cnn_y = {
    s: np.load(FEATURE_DIR / f"cnn_y_{s}.npy")
    for s in splits
}

sift_y = {
    s: np.load(FEATURE_DIR / f"y_{s}.npy")
    for s in splits
}

# Verify sample counts and label alignment
for s in splits:
    assert len(cnn[s]) == len(sift[s])
    assert np.array_equal(cnn_y[s], sift_y[s]), (
        f"Label mismatch in {s}"
    )

    print(
        s,
        "CNN:", cnn[s].shape,
        "SIFT:", sift[s].shape,
        "Samples:", len(cnn_y[s])
    )

# Fit scalers using training data only
cnn_scaler = StandardScaler()
sift_scaler = StandardScaler()

cnn_scaled = {
    "train": cnn_scaler.fit_transform(cnn["train"]),
    "val": cnn_scaler.transform(cnn["val"]),
    "test": cnn_scaler.transform(cnn["test"])
}

sift_scaled = {
    "train": sift_scaler.fit_transform(sift["train"]),
    "val": sift_scaler.transform(sift["val"]),
    "test": sift_scaler.transform(sift["test"])
}

y_train = cnn_y["train"]
y_val = cnn_y["val"]
y_test = cnn_y["test"]

# Hyperparameter search on validation set
sift_weights = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
C_values = [0.1, 1.0, 10.0]

results = []

for C in C_values:
    for beta in sift_weights:

        X_train = np.concatenate(
            [cnn_scaled["train"], beta * sift_scaled["train"]],
            axis=1
        )

        X_val = np.concatenate(
            [cnn_scaled["val"], beta * sift_scaled["val"]],
            axis=1
        )

        model = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="lbfgs"
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        acc = accuracy_score(y_val, pred)
        macro_f1 = f1_score(
            y_val, pred, average="macro", zero_division=0
        )

        results.append({
            "C": C,
            "sift_weight": beta,
            "val_accuracy": acc,
            "val_macro_f1": macro_f1
        })

        print(
            f"C={C} beta={beta} "
            f"Val Acc={acc:.4%} "
            f"Val Macro-F1={macro_f1:.4%}"
        )

# Select best configuration using validation Macro-F1
results_df = pd.DataFrame(results).sort_values(
    ["val_macro_f1", "val_accuracy"],
    ascending=False
)

results_df.to_csv(
    OUT_DIR / "validation_fusion_results.csv",
    index=False
)

best = results_df.iloc[0]
best_C = float(best["C"])
best_beta = float(best["sift_weight"])

print("\nSelected configuration:")
print(best)

# Train final weighted hybrid
X_train_best = np.concatenate(
    [cnn_scaled["train"], best_beta * sift_scaled["train"]],
    axis=1
)

X_test_best = np.concatenate(
    [cnn_scaled["test"], best_beta * sift_scaled["test"]],
    axis=1
)

hybrid_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    solver="lbfgs"
)

hybrid_model.fit(X_train_best, y_train)
hybrid_pred = hybrid_model.predict(X_test_best)

hybrid_acc = accuracy_score(y_test, hybrid_pred)
hybrid_f1 = f1_score(
    y_test, hybrid_pred, average="macro", zero_division=0
)

# Matched CNN-only baseline
cnn_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    solver="lbfgs"
)

cnn_model.fit(cnn_scaled["train"], y_train)
cnn_pred = cnn_model.predict(cnn_scaled["test"])

cnn_acc = accuracy_score(y_test, cnn_pred)
cnn_f1 = f1_score(
    y_test, cnn_pred, average="macro", zero_division=0
)

# Matched SIFT-only baseline
sift_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    solver="lbfgs"
)

sift_model.fit(sift_scaled["train"], y_train)
sift_pred = sift_model.predict(sift_scaled["test"])

sift_acc = accuracy_score(y_test, sift_pred)
sift_f1 = f1_score(
    y_test, sift_pred, average="macro", zero_division=0
)

print("\n=== CALTECH-101 TEST COMPARISON ===")
print(f"CNN-only:  {cnn_acc:.4%} | {cnn_f1:.4%}")
print(f"SIFT-only: {sift_acc:.4%} | {sift_f1:.4%}")
print(f"Weighted:  {hybrid_acc:.4%} | {hybrid_f1:.4%}")

print("\n=== HYBRID DELTA VS CNN ===")
print(f"Accuracy: {(hybrid_acc-cnn_acc)*100:+.2f} pp")
print(f"Macro-F1: {(hybrid_f1-cnn_f1)*100:+.2f} pp")

# Save predictions
np.save(OUT_DIR / "cnn_only_predictions.npy", cnn_pred)
np.save(OUT_DIR / "sift_only_predictions.npy", sift_pred)
np.save(OUT_DIR / "weighted_hybrid_predictions.npy", hybrid_pred)

# Save classification report
report = classification_report(
    y_test, hybrid_pred,
    output_dict=True,
    zero_division=0
)

with open(OUT_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Save complete summary
summary = {
    "selected_C": best_C,
    "selected_sift_weight": best_beta,
    "validation_accuracy": float(best["val_accuracy"]),
    "validation_macro_f1": float(best["val_macro_f1"]),
    "cnn_only_accuracy": float(cnn_acc),
    "cnn_only_macro_f1": float(cnn_f1),
    "sift_only_accuracy": float(sift_acc),
    "sift_only_macro_f1": float(sift_f1),
    "weighted_hybrid_accuracy": float(hybrid_acc),
    "weighted_hybrid_macro_f1": float(hybrid_f1),
    "accuracy_delta_pp": float((hybrid_acc-cnn_acc)*100),
    "macro_f1_delta_pp": float((hybrid_f1-cnn_f1)*100)
}

with open(OUT_DIR / "final_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved results to:", OUT_DIR)

