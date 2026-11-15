
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]

CNN_DIR = ROOT / "results_caltech256" / "hybrid_features"
SIFT_DIR = ROOT / "results_caltech256" / "sift_bovw_svm"
OUT_DIR = ROOT / "results_caltech256" / "weighted_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features(directory, prefix, splits):
    return {
        split: np.load(directory / f"{prefix}_{split}.npy")
        for split in splits
    }


splits = ["train", "val", "test"]

# Load CNN features and labels
cnn = load_features(CNN_DIR, "cnn", splits)
cnn_y = {
    s: np.load(CNN_DIR / f"cnn_y_{s}.npy")
    for s in splits
}

# Load SIFT-BoVW features and labels
sift = load_features(SIFT_DIR, "X", splits)
sift_y = {
    s: np.load(SIFT_DIR / f"y_{s}.npy")
    for s in splits
}

# Verify feature dimensions and sample counts
for s in splits:
    assert cnn[s].shape[0] == sift[s].shape[0], (
        f"{s}: CNN and SIFT sample counts differ"
    )
    assert len(cnn_y[s]) == len(sift_y[s])
    assert np.array_equal(cnn_y[s], sift_y[s]), (
        f"{s}: CNN and SIFT labels/order do not match"
    )

    print(
        s,
        "CNN:", cnn[s].shape,
        "SIFT:", sift[s].shape,
        "Samples:", len(cnn_y[s])
    )

# Fit separate scalers ONLY on training data
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

# Baselines and fusion configurations
sift_weights = [0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
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
            f"C={C:<4} beta={beta:<4} "
            f"Val Acc={acc:.4%} "
            f"Val Macro-F1={macro_f1:.4%}"
        )

# Save all validation results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["val_macro_f1", "val_accuracy"],
    ascending=False
)

results_df.to_csv(
    OUT_DIR / "validation_fusion_results.csv",
    index=False
)

# Select using validation Macro-F1, then accuracy
best = results_df.iloc[0]

best_C = float(best["C"])
best_beta = float(best["sift_weight"])

print("\nSelected configuration:")
print(best)

# Refit selected model using training data only
X_train_best = np.concatenate(
    [cnn_scaled["train"], best_beta * sift_scaled["train"]],
    axis=1
)

X_test_best = np.concatenate(
    [cnn_scaled["test"], best_beta * sift_scaled["test"]],
    axis=1
)

final_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    solver="lbfgs"
)

final_model.fit(X_train_best, y_train)
test_pred = final_model.predict(X_test_best)

test_accuracy = accuracy_score(y_test, test_pred)
test_macro_f1 = f1_score(
    y_test, test_pred, average="macro", zero_division=0
)

print("\nFINAL HELD-OUT TEST RESULTS")
print(f"Accuracy: {test_accuracy:.4%}")
print(f"Macro-F1: {test_macro_f1:.4%}")


# FAIR COMPARISON: CNN-only
cnn_only_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    solver="lbfgs"
)

cnn_only_model.fit(cnn_scaled["train"], y_train)
cnn_only_pred = cnn_only_model.predict(cnn_scaled["test"])

cnn_only_acc = accuracy_score(y_test, cnn_only_pred)
cnn_only_f1 = f1_score(
    y_test, cnn_only_pred, average="macro", zero_division=0
)

# FAIR COMPARISON: SIFT-only
sift_only_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    solver="lbfgs"
)

sift_only_model.fit(sift_scaled["train"], y_train)
sift_only_pred = sift_only_model.predict(sift_scaled["test"])

sift_only_acc = accuracy_score(y_test, sift_only_pred)
sift_only_f1 = f1_score(
    y_test, sift_only_pred, average="macro", zero_division=0
)

print("\n=== FAIR TEST COMPARISON ===")
print(f"CNN-only:  {cnn_only_acc:.4%} | {cnn_only_f1:.4%}")
print(f"SIFT-only: {sift_only_acc:.4%} | {sift_only_f1:.4%}")
print(f"Hybrid:    {test_accuracy:.4%} | {test_macro_f1:.4%}")

print("\n=== HYBRID DELTA VS CNN ===")
print(f"Accuracy: {(test_accuracy-cnn_only_acc)*100:+.2f} pp")
print(f"Macro-F1: {(test_macro_f1-cnn_only_f1)*100:+.2f} pp")
print("\nSaved results to:", OUT_DIR)

# Save predictions and report

np.save(OUT_DIR / "weighted_hybrid_test_predictions.npy", test_pred)
np.save(OUT_DIR / "cnn_only_test_predictions.npy", cnn_only_pred)
np.save(OUT_DIR / "sift_only_test_predictions.npy", sift_only_pred)

summary = {
    "selected_C": best_C,
    "selected_sift_weight": best_beta,
    "validation_accuracy": float(best["val_accuracy"]),
    "validation_macro_f1": float(best["val_macro_f1"]),
    "weighted_hybrid_accuracy": float(test_accuracy),
    "weighted_hybrid_macro_f1": float(test_macro_f1),
    "cnn_only_accuracy": float(cnn_only_acc),
    "cnn_only_macro_f1": float(cnn_only_f1),
    "sift_only_accuracy": float(sift_only_acc),
    "sift_only_macro_f1": float(sift_only_f1)
}

with open(OUT_DIR / "final_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved results to:", OUT_DIR)
