import os
import json
import time
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


RESULTS_DIR = "results/sift"

# Existing BoVW features
X_train = np.load("results/hybrid/sift_train.npy")
X_val = np.load("results/hybrid/sift_val.npy")
X_test = np.load("results/hybrid/sift_test.npy")

y_train = np.load("results/hybrid/y_train.npy")
y_val = np.load("results/hybrid/y_val.npy")
y_test = np.load("results/hybrid/y_test.npy")


# ---------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------

C_VALUES = [1, 10, 100]
GAMMA_VALUES = ["scale", 0.001, 0.003, 0.01]

results = []

best_f1 = -1
best_model = None
best_params = None


print("=" * 70)
print("SIFT RBF-SVM HYPERPARAMETER SEARCH")
print("=" * 70)

print(f"Train shape: {X_train.shape}")
print(f"Val shape:   {X_val.shape}")
print(f"Test shape:  {X_test.shape}")

total = len(C_VALUES) * len(GAMMA_VALUES)
run_number = 0


for C in C_VALUES:
    for gamma in GAMMA_VALUES:

        run_number += 1

        print("\n" + "-" * 70)
        print(f"Run {run_number}/{total}")
        print(f"C = {C}, gamma = {gamma}")
        print("-" * 70)

        start = time.time()

        clf = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            cache_size=4096
        )

        clf.fit(X_train, y_train)

        val_pred = clf.predict(X_val)

        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro"
        )

        elapsed = time.time() - start

        print(f"Validation Accuracy : {val_acc * 100:.2f}%")
        print(f"Validation Macro-F1 : {val_f1 * 100:.2f}%")
        print(f"Time                : {elapsed / 60:.2f} min")

        results.append({
            "C": C,
            "gamma": gamma,
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
            "time_seconds": float(elapsed)
        })

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_model = clf
            best_params = {
                "C": C,
                "gamma": gamma
            }

            print(">>> NEW BEST MODEL")


# ---------------------------------------------------------
# Print ranking
# ---------------------------------------------------------

results_sorted = sorted(
    results,
    key=lambda x: x["val_macro_f1"],
    reverse=True
)

print("\n" + "=" * 70)
print("VALIDATION RANKING")
print("=" * 70)

for i, r in enumerate(results_sorted, 1):

    print(
        f"{i:2d}. "
        f"C={r['C']:>3}, "
        f"gamma={str(r['gamma']):>6} | "
        f"Acc={r['val_accuracy'] * 100:6.2f}% | "
        f"Macro-F1={r['val_macro_f1'] * 100:6.2f}%"
    )


# ---------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODEL — FINAL TEST EVALUATION")
print("=" * 70)

print(f"Best C     : {best_params['C']}")
print(f"Best gamma : {best_params['gamma']}")

test_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, test_pred)

test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)

print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Macro-F1 : {test_f1 * 100:.2f}%")


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

np.save(
    os.path.join(
        RESULTS_DIR,
        "tuned_svm_test_predictions.npy"
    ),
    test_pred
)

with open(
    os.path.join(
        RESULTS_DIR,
        "tuned_svm_classification_report.txt"
    ),
    "w"
) as f:

    f.write(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        )
    )

summary = {
    "search_space": {
        "C": C_VALUES,
        "gamma": GAMMA_VALUES
    },
    "selection_metric": "validation_macro_f1",
    "best_parameters": best_params,
    "best_validation_macro_f1": float(best_f1),
    "best_validation_accuracy": float(
        results_sorted[0]["val_accuracy"]
    ),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_f1),
    "all_results": results_sorted
}

with open(
    os.path.join(
        RESULTS_DIR,
        "tuned_svm_results.json"
    ),
    "w"
) as f:

    json.dump(summary, f, indent=2)


print("\nSaved:")
print("  results/sift/tuned_svm_test_predictions.npy")
print("  results/sift/tuned_svm_classification_report.txt")
print("  results/sift/tuned_svm_results.json")

print("\nTuning complete.")
