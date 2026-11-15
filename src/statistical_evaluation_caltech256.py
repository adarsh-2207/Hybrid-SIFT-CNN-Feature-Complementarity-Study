
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# CONFIGURATION
# =========================================================

BASE = Path("results_caltech256")
SOURCE = BASE / "pca_hybrid_fusion"
OUT = BASE / "statistical_evaluation"

OUT.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 5000
SEED = 42

# =========================================================
# LOAD PAIRED TEST PREDICTIONS
# =========================================================

df = pd.read_csv(
    SOURCE / "test_predictions.csv"
)

y = df["true_label"].to_numpy()
cnn = df["cnn_prediction"].to_numpy()
hybrid = df["hybrid_prediction"].to_numpy()

assert len(y) == len(cnn) == len(hybrid)
assert len(y) == 4472

cnn_correct = cnn == y
hybrid_correct = hybrid == y

print("Test samples:", len(y))


# =========================================================
# BASIC METRICS
# =========================================================

def metrics(y_true, pred):

    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0
        )
    }


cnn_metrics = metrics(y, cnn)
hybrid_metrics = metrics(y, hybrid)

print("\nCNN-only:", cnn_metrics)
print("PCA Hybrid:", hybrid_metrics)


# =========================================================
# MCNEMAR'S EXACT TEST
# =========================================================

cnn_correct_hybrid_wrong = int(
    (cnn_correct & ~hybrid_correct).sum()
)

cnn_wrong_hybrid_correct = int(
    (~cnn_correct & hybrid_correct).sum()
)

both_correct = int(
    (cnn_correct & hybrid_correct).sum()
)

both_wrong = int(
    (~cnn_correct & ~hybrid_correct).sum()
)

discordant = (
    cnn_correct_hybrid_wrong
    + cnn_wrong_hybrid_correct
)

if discordant > 0:

    mcnemar = binomtest(
        cnn_correct_hybrid_wrong,
        n=discordant,
        p=0.5,
        alternative="two-sided"
    )

    p_value = float(mcnemar.pvalue)

else:
    p_value = 1.0

print("\n" + "=" * 60)
print("MCNEMAR'S EXACT TEST")
print("=" * 60)

print("CNN correct, hybrid wrong:", cnn_correct_hybrid_wrong)
print("CNN wrong, hybrid correct:", cnn_wrong_hybrid_correct)
print("Both correct:", both_correct)
print("Both wrong:", both_wrong)
print("Discordant pairs:", discordant)
print("Exact p-value:", p_value)


# =========================================================
# CLASS-STRATIFIED PAIRED BOOTSTRAP
# =========================================================

print("\nRunning paired bootstrap...")

rng = np.random.default_rng(SEED)

classes = np.unique(y)

class_indices = {
    c: np.where(y == c)[0]
    for c in classes
}

accuracy_differences = []
macro_f1_differences = []

for iteration in range(N_BOOTSTRAP):

    # Resample independently within each class.
    # Both models use exactly the same sampled indices.

    sampled_indices = np.concatenate([
        rng.choice(
            class_indices[c],
            size=len(class_indices[c]),
            replace=True
        )
        for c in classes
    ])

    y_sample = y[sampled_indices]

    cnn_sample = cnn[sampled_indices]
    hybrid_sample = hybrid[sampled_indices]

    cnn_acc = accuracy_score(
        y_sample, cnn_sample
    )

    hybrid_acc = accuracy_score(
        y_sample, hybrid_sample
    )

    cnn_f1 = f1_score(
        y_sample,
        cnn_sample,
        average="macro",
        zero_division=0
    )

    hybrid_f1 = f1_score(
        y_sample,
        hybrid_sample,
        average="macro",
        zero_division=0
    )

    accuracy_differences.append(
        hybrid_acc - cnn_acc
    )

    macro_f1_differences.append(
        hybrid_f1 - cnn_f1
    )


accuracy_differences = np.array(
    accuracy_differences
)

macro_f1_differences = np.array(
    macro_f1_differences
)


# =========================================================
# CONFIDENCE INTERVALS
# =========================================================

def confidence_interval(values):

    return np.percentile(
        values,
        [2.5, 97.5]
    )


acc_ci = confidence_interval(
    accuracy_differences
)

f1_ci = confidence_interval(
    macro_f1_differences
)

observed_acc_diff = (
    hybrid_metrics["accuracy"]
    - cnn_metrics["accuracy"]
)

observed_f1_diff = (
    hybrid_metrics["macro_f1"]
    - cnn_metrics["macro_f1"]
)

print("\n" + "=" * 60)
print("PAIRED BOOTSTRAP RESULTS")
print("=" * 60)

print(
    f"Accuracy difference: {observed_acc_diff:.6f}"
)

print(
    f"Accuracy 95% CI: [{acc_ci[0]:.6f}, {acc_ci[1]:.6f}]"
)

print(
    f"Macro-F1 difference: {observed_f1_diff:.6f}"
)

print(
    f"Macro-F1 95% CI: [{f1_ci[0]:.6f}, {f1_ci[1]:.6f}]"
)


# =========================================================
# SAVE RESULTS
# =========================================================

summary = {
    "test_samples": len(y),
    "cnn_accuracy": cnn_metrics["accuracy"],
    "hybrid_accuracy": hybrid_metrics["accuracy"],
    "cnn_macro_f1": cnn_metrics["macro_f1"],
    "hybrid_macro_f1": hybrid_metrics["macro_f1"],
    "accuracy_difference_hybrid_minus_cnn": observed_acc_diff,
    "accuracy_difference_ci_lower": float(acc_ci[0]),
    "accuracy_difference_ci_upper": float(acc_ci[1]),
    "macro_f1_difference_hybrid_minus_cnn": observed_f1_diff,
    "macro_f1_difference_ci_lower": float(f1_ci[0]),
    "macro_f1_difference_ci_upper": float(f1_ci[1]),
    "mcnemar_exact_p_value": p_value,
    "cnn_correct_hybrid_wrong": cnn_correct_hybrid_wrong,
    "cnn_wrong_hybrid_correct": cnn_wrong_hybrid_correct,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "bootstrap_replicates": N_BOOTSTRAP,
    "bootstrap_method": "class-stratified paired bootstrap"
}

pd.DataFrame([summary]).to_csv(
    OUT / "statistical_summary.csv",
    index=False
)

import json

with open(OUT / "statistical_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nResults saved in:", OUT)

if p_value < 0.05:
    print("McNemar: statistically significant difference.")
else:
    print("McNemar: no statistically significant difference detected.")

