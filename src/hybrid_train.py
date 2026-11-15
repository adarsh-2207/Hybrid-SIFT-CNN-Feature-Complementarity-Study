import os
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

FEATURE_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid"
)

RESULT_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid"
)

os.makedirs(RESULT_ROOT, exist_ok=True)


# =========================================================
# Load features
# =========================================================

print("Loading features...")

X_sift_train = np.load(
    os.path.join(FEATURE_ROOT, "sift_train.npy")
)

X_sift_val = np.load(
    os.path.join(FEATURE_ROOT, "sift_val.npy")
)

X_sift_test = np.load(
    os.path.join(FEATURE_ROOT, "sift_test.npy")
)

X_cnn_train = np.load(
    os.path.join(FEATURE_ROOT, "cnn_train.npy")
)

X_cnn_val = np.load(
    os.path.join(FEATURE_ROOT, "cnn_val.npy")
)

X_cnn_test = np.load(
    os.path.join(FEATURE_ROOT, "cnn_test.npy")
)

y_train = np.load(
    os.path.join(FEATURE_ROOT, "y_train.npy")
)

y_val = np.load(
    os.path.join(FEATURE_ROOT, "y_val.npy")
)

y_test = np.load(
    os.path.join(FEATURE_ROOT, "y_test.npy")
)


print("SIFT train:", X_sift_train.shape)
print("CNN train:", X_cnn_train.shape)


# =========================================================
# Verify labels
# =========================================================

assert np.array_equal(
    y_train,
    np.load(os.path.join(FEATURE_ROOT, "cnn_y_train.npy"))
)

assert np.array_equal(
    y_val,
    np.load(os.path.join(FEATURE_ROOT, "cnn_y_val.npy"))
)

assert np.array_equal(
    y_test,
    np.load(os.path.join(FEATURE_ROOT, "cnn_y_test.npy"))
)

print("Label alignment: OK")


# =========================================================
# Standardize CNN features
# =========================================================

print("\nStandardizing CNN features...")

cnn_scaler = StandardScaler()

X_cnn_train_scaled = cnn_scaler.fit_transform(
    X_cnn_train
)

X_cnn_val_scaled = cnn_scaler.transform(
    X_cnn_val
)

X_cnn_test_scaled = cnn_scaler.transform(
    X_cnn_test
)


# =========================================================
# Standardize SIFT features
# =========================================================

print("Standardizing SIFT features...")

sift_scaler = StandardScaler()

X_sift_train_scaled = sift_scaler.fit_transform(
    X_sift_train
)

X_sift_val_scaled = sift_scaler.transform(
    X_sift_val
)

X_sift_test_scaled = sift_scaler.transform(
    X_sift_test
)


# =========================================================
# Create hybrid representation
# =========================================================

print("\nCreating hybrid representation...")

X_hybrid_train = np.concatenate(
    [
        X_cnn_train_scaled,
        X_sift_train_scaled
    ],
    axis=1
)

X_hybrid_val = np.concatenate(
    [
        X_cnn_val_scaled,
        X_sift_val_scaled
    ],
    axis=1
)

X_hybrid_test = np.concatenate(
    [
        X_cnn_test_scaled,
        X_sift_test_scaled
    ],
    axis=1
)

print("Hybrid train:", X_hybrid_train.shape)
print("Hybrid val:", X_hybrid_val.shape)
print("Hybrid test:", X_hybrid_test.shape)


# =========================================================
# Train classifier
# =========================================================

def train_and_evaluate(
    name,
    X_train,
    X_val,
    X_test
):

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

  
    classifier = LogisticRegression(
    max_iter=2000,
    C=1.0,
    solver="lbfgs",
    random_state=42
)
    classifier.fit(
        X_train,
        y_train
    )

    val_predictions = classifier.predict(X_val)

    test_predictions = classifier.predict(X_test)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro"
    )

    print(f"\n{name} Results")

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Macro-F1: "
        f"{val_macro_f1 * 100:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro-F1: "
        f"{test_macro_f1 * 100:.2f}%"
    )

    report = classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )

    with open(
        os.path.join(
            RESULT_ROOT,
            f"{name.lower()}_classification_report.txt"
        ),
        "w"
    ) as f:
        f.write(report)

    metrics = {
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(val_macro_f1),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1)
    }

    with open(
        os.path.join(
            RESULT_ROOT,
            f"{name.lower()}_metrics.json"
        ),
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    return metrics


# =========================================================
# CNN-only matched classifier
# =========================================================

cnn_results = train_and_evaluate(
    "CNN_Matched",
    X_cnn_train_scaled,
    X_cnn_val_scaled,
    X_cnn_test_scaled
)


# =========================================================
# Hybrid classifier
# =========================================================

hybrid_results = train_and_evaluate(
    "Hybrid",
    X_hybrid_train,
    X_hybrid_val,
    X_hybrid_test
)


# =========================================================
# Final comparison
# =========================================================

accuracy_gain = (
    hybrid_results["test_accuracy"]
    - cnn_results["test_accuracy"]
)

f1_gain = (
    hybrid_results["test_macro_f1"]
    - cnn_results["test_macro_f1"]
)

print("\n")
print("=" * 60)
print("FINAL CNN vs HYBRID COMPARISON")
print("=" * 60)

print(
    f"CNN Test Accuracy:     "
    f"{cnn_results['test_accuracy'] * 100:.2f}%"
)

print(
    f"Hybrid Test Accuracy:  "
    f"{hybrid_results['test_accuracy'] * 100:.2f}%"
)

print(
    f"Accuracy Gain:         "
    f"{accuracy_gain * 100:+.2f} pp"
)

print()

print(
    f"CNN Test Macro-F1:     "
    f"{cnn_results['test_macro_f1'] * 100:.2f}%"
)

print(
    f"Hybrid Test Macro-F1:  "
    f"{hybrid_results['test_macro_f1'] * 100:.2f}%"
)

print(
    f"Macro-F1 Gain:         "
    f"{f1_gain * 100:+.2f} pp"
)

print("=" * 60)
