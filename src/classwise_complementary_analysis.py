
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.special import softmax

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE = Path("results_caltech256")
SOURCE = BASE / "late_fusion_mlp"
OUT = BASE / "classwise_analysis"

OUT.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 256

# Selected in previous validation experiments
RAW_ALPHA = 0.75
SCALED_ALPHA = 0.80

# =========================================================
# LOAD PROBABILITIES AND LABELS
# =========================================================

print("Loading saved predictions...")

cnn_val = np.load(SOURCE / "cnn_val_probabilities.npy")
sift_val = np.load(SOURCE / "sift_val_probabilities.npy")

cnn_test = np.load(SOURCE / "cnn_test_probabilities.npy")
sift_test = np.load(SOURCE / "sift_test_probabilities.npy")

y_val = np.load(
    BASE / "hybrid_features" / "cnn_y_val.npy"
).astype(int)

y_test = np.load(
    BASE / "hybrid_features" / "cnn_y_test.npy"
).astype(int)

sift_y_val = np.load(
    BASE / "sift_bovw_svm" / "y_val.npy"
).astype(int)

sift_y_test = np.load(
    BASE / "sift_bovw_svm" / "y_test.npy"
).astype(int)

assert np.array_equal(y_val, sift_y_val)
assert np.array_equal(y_test, sift_y_test)

assert cnn_test.shape == sift_test.shape
assert cnn_test.shape == (len(y_test), NUM_CLASSES)

print("Test samples:", len(y_test))
print("Classes:", NUM_CLASSES)


# =========================================================
# TEMPERATURE SCALING
# Fit temperatures using validation data ONLY
# =========================================================

def fit_temperature(probabilities, labels):

    def objective(temp):

        calibrated = softmax(
            np.log(np.clip(probabilities, 1e-12, 1.0)) / temp,
            axis=1
        )

        selected = calibrated[
            np.arange(len(labels)), labels
        ]

        return -np.mean(np.log(np.clip(selected, 1e-12, 1.0)))

    result = minimize_scalar(
        objective,
        bounds=(0.05, 10.0),
        method="bounded"
    )

    return float(result.x)


def apply_temperature(probabilities, temperature):

    logits = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )

    return softmax(logits / temperature, axis=1)


cnn_temp = fit_temperature(cnn_val, y_val)
sift_temp = fit_temperature(sift_val, y_val)

print("\nValidation-fitted temperatures:")
print("CNN:", round(cnn_temp, 5))
print("SIFT:", round(sift_temp, 5))

cnn_test_cal = apply_temperature(cnn_test, cnn_temp)
sift_test_cal = apply_temperature(sift_test, sift_temp)


# =========================================================
# GENERATE FOUR MODEL PREDICTIONS
# =========================================================

cnn_pred = cnn_test.argmax(axis=1)
sift_pred = sift_test.argmax(axis=1)

raw_fusion_probs = (
    RAW_ALPHA * cnn_test
    + (1 - RAW_ALPHA) * sift_test
)

# Normalize probability rows
raw_fusion_probs /= raw_fusion_probs.sum(
    axis=1, keepdims=True
)

raw_fusion_pred = raw_fusion_probs.argmax(axis=1)

scaled_fusion_probs = (
    SCALED_ALPHA * cnn_test_cal
    + (1 - SCALED_ALPHA) * sift_test_cal
)

# Normalize probability rows
scaled_fusion_probs /= scaled_fusion_probs.sum(
    axis=1, keepdims=True
)

scaled_fusion_pred = scaled_fusion_probs.argmax(axis=1)

predictions = {
    "CNN": cnn_pred,
    "SIFT": sift_pred,
    "Raw_Late_Fusion": raw_fusion_pred,
    "Temperature_Scaled_Fusion": scaled_fusion_pred
}


# =========================================================
# OVERALL METRICS
# =========================================================

print("\n" + "=" * 65)
print("OVERALL TEST PERFORMANCE")
print("=" * 65)

overall_rows = []

for name, pred in predictions.items():

    overall_rows.append({
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

overall_df = pd.DataFrame(overall_rows)

print(overall_df.to_string(index=False))

overall_df.to_csv(
    OUT / "overall_metrics.csv",
    index=False
)


# =========================================================
# ERROR OVERLAP ANALYSIS
# =========================================================

cnn_correct = cnn_pred == y_test
sift_correct = sift_pred == y_test

raw_fusion_correct = raw_fusion_pred == y_test
scaled_fusion_correct = scaled_fusion_pred == y_test

patterns = {
    "CNN_correct_SIFT_correct": cnn_correct & sift_correct,
    "CNN_correct_SIFT_wrong": cnn_correct & ~sift_correct,
    "CNN_wrong_SIFT_correct": ~cnn_correct & sift_correct,
    "CNN_wrong_SIFT_wrong": ~cnn_correct & ~sift_correct
}

print("\n" + "=" * 65)
print("CNN VS SIFT ERROR OVERLAP")
print("=" * 65)

overlap_rows = []

for name, mask in patterns.items():

    count = int(mask.sum())

    overlap_rows.append({
        "pattern": name,
        "count": count,
        "percentage": 100 * count / len(y_test)
    })

    print(
        f"{name}: {count} "
        f"({100 * count / len(y_test):.2f}%)"
    )

overlap_df = pd.DataFrame(overlap_rows)

overlap_df.to_csv(
    OUT / "cnn_sift_error_overlap.csv",
    index=False
)


# =========================================================
# DOES FUSION FIX CNN ERRORS?
# =========================================================

print("\n" + "=" * 65)
print("FUSION ERROR CORRECTION ANALYSIS")
print("=" * 65)

fusion_rows = []

for name, pred in [
    ("Raw_Late_Fusion", raw_fusion_pred),
    ("Temperature_Scaled_Fusion", scaled_fusion_pred)
]:

    fusion_correct = pred == y_test

    corrected = (~cnn_correct) & fusion_correct
    damaged = cnn_correct & (~fusion_correct)

    unchanged_correct = cnn_correct & fusion_correct
    unchanged_wrong = (~cnn_correct) & (~fusion_correct)

    row = {
        "fusion_model": name,
        "cnn_errors_corrected": int(corrected.sum()),
        "cnn_correct_predictions_damaged": int(damaged.sum()),
        "both_correct": int(unchanged_correct.sum()),
        "both_wrong": int(unchanged_wrong.sum()),
        "net_correct_change": int(
            corrected.sum() - damaged.sum()
        )
    }

    fusion_rows.append(row)

    print("\n", name)
    for key, value in row.items():
        if key != "fusion_model":
            print(f"{key}: {value}")

fusion_df = pd.DataFrame(fusion_rows)

fusion_df.to_csv(
    OUT / "fusion_error_correction.csv",
    index=False
)


# =========================================================
# CLASS-WISE METRICS
# =========================================================

print("\nCalculating class-wise metrics...")

class_rows = []

for class_id in range(NUM_CLASSES):

    class_mask = y_test == class_id

    true_class_count = int(class_mask.sum())

    if true_class_count == 0:
        continue

    row = {
        "class_id": class_id,
        "test_support": true_class_count
    }

    for name, pred in predictions.items():

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                pred,
                labels=[class_id],
                average=None,
                zero_division=0
            )
        )

        row[f"{name}_precision"] = precision[0]
        row[f"{name}_recall"] = recall[0]
        row[f"{name}_f1"] = f1[0]

    # CNN versus SIFT
    row["sift_minus_cnn_f1"] = (
        row["SIFT_f1"] - row["CNN_f1"]
    )

    # Raw fusion versus CNN
    row["raw_fusion_minus_cnn_f1"] = (
        row["Raw_Late_Fusion_f1"] - row["CNN_f1"]
    )

    # Calibrated fusion versus CNN
    row["scaled_fusion_minus_cnn_f1"] = (
        row["Temperature_Scaled_Fusion_f1"]
        - row["CNN_f1"]
    )

    # Number of samples in this class where SIFT rescues CNN
    row["sift_rescues_cnn_count"] = int(
        (
            class_mask
            & ~cnn_correct
            & sift_correct
        ).sum()
    )

    # Number of samples where SIFT is correct but raw fusion is wrong
    row["raw_fusion_loses_sift_correct_count"] = int(
        (
            class_mask
            & sift_correct
            & ~raw_fusion_correct
        ).sum()
    )

    class_rows.append(row)


class_df = pd.DataFrame(class_rows)

class_df.to_csv(
    OUT / "classwise_metrics.csv",
    index=False
)


# =========================================================
# TOP CLASS-WISE IMPROVEMENTS AND DECLINES
# =========================================================

def save_ranked_classes(column, filename, ascending=False):

    ranked = class_df.sort_values(
        column,
        ascending=ascending
    )

    ranked.head(20).to_csv(
        OUT / filename,
        index=False
    )

    return ranked.head(20)


print("\n" + "=" * 65)
print("TOP 20 CLASSES: RAW FUSION IMPROVEMENT OVER CNN")
print("=" * 65)

top_improvements = save_ranked_classes(
    "raw_fusion_minus_cnn_f1",
    "top_20_raw_fusion_improvements.csv"
)

print(
    top_improvements[
        [
            "class_id",
            "test_support",
            "CNN_f1",
            "Raw_Late_Fusion_f1",
            "raw_fusion_minus_cnn_f1"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 65)
print("TOP 20 CLASSES: RAW FUSION DECLINES")
print("=" * 65)

top_declines = save_ranked_classes(
    "raw_fusion_minus_cnn_f1",
    "top_20_raw_fusion_declines.csv",
    ascending=True
)

print(
    top_declines[
        [
            "class_id",
            "test_support",
            "CNN_f1",
            "Raw_Late_Fusion_f1",
            "raw_fusion_minus_cnn_f1"
        ]
    ].to_string(index=False)
)


# =========================================================
# SIFT COMPLEMENTARY CLASS ANALYSIS
# =========================================================

print("\n" + "=" * 65)
print("CLASSES WHERE SIFT OUTPERFORMS CNN")
print("=" * 65)

sift_better = class_df[
    class_df["sift_minus_cnn_f1"] > 0
].sort_values(
    "sift_minus_cnn_f1",
    ascending=False
)

print("Number of classes where SIFT F1 > CNN F1:",
      len(sift_better))

sift_better.head(20).to_csv(
    OUT / "top_20_sift_better_classes.csv",
    index=False
)

print(
    sift_better[
        [
            "class_id",
            "test_support",
            "CNN_f1",
            "SIFT_f1",
            "sift_minus_cnn_f1",
            "sift_rescues_cnn_count"
        ]
    ].head(20).to_string(index=False)
)


# =========================================================
# SAVE SAMPLE-LEVEL PREDICTIONS
# =========================================================

sample_df = pd.DataFrame({
    "true_label": y_test,
    "cnn_prediction": cnn_pred,
    "sift_prediction": sift_pred,
    "raw_fusion_prediction": raw_fusion_pred,
    "scaled_fusion_prediction": scaled_fusion_pred,
    "cnn_correct": cnn_correct,
    "sift_correct": sift_correct,
    "raw_fusion_correct": raw_fusion_correct,
    "scaled_fusion_correct": scaled_fusion_correct
})

sample_df.to_csv(
    OUT / "sample_level_predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)

print("Saved in:", OUT)

