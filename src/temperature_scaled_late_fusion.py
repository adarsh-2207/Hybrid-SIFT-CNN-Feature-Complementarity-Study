
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE = Path("results_caltech256")
SOURCE = BASE / "late_fusion_mlp"
OUT = BASE / "temperature_scaled_late_fusion"

OUT.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 256
ALPHAS = np.round(np.arange(0.0, 1.001, 0.05), 2)

# =========================================================
# LOAD PREVIOUS EXPERIMENT PROBABILITIES
# =========================================================

print("Loading saved probabilities...")

cnn_val = np.load(
    SOURCE / "cnn_val_probabilities.npy"
)

sift_val = np.load(
    SOURCE / "sift_val_probabilities.npy"
)

cnn_test = np.load(
    SOURCE / "cnn_test_probabilities.npy"
)

sift_test = np.load(
    SOURCE / "sift_test_probabilities.npy"
)

cnn_val_labels = np.load(
    BASE / "hybrid_features" / "cnn_y_val.npy"
).astype(int)

sift_val_labels = np.load(
    BASE / "sift_bovw_svm" / "y_val.npy"
).astype(int)

cnn_test_labels = np.load(
    BASE / "hybrid_features" / "cnn_y_test.npy"
).astype(int)

sift_test_labels = np.load(
    BASE / "sift_bovw_svm" / "y_test.npy"
).astype(int)

assert np.array_equal(cnn_val_labels, sift_val_labels)
assert np.array_equal(cnn_test_labels, sift_test_labels)

y_val = cnn_val_labels
y_test = cnn_test_labels

assert cnn_val.shape == sift_val.shape
assert cnn_test.shape == sift_test.shape
assert cnn_val.shape[1] == NUM_CLASSES

print("CNN validation:", cnn_val.shape)
print("SIFT validation:", sift_val.shape)
print("CNN test:", cnn_test.shape)
print("SIFT test:", sift_test.shape)


# =========================================================
# TEMPERATURE SCALING
# =========================================================

def apply_temperature(probabilities, temperature):

    # Recover logits up to an irrelevant per-sample constant.
    logits = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )

    return softmax(
        logits / temperature,
        axis=1
    )


def fit_temperature(probabilities, labels):

    def objective(temperature):

        calibrated = apply_temperature(
            probabilities,
            temperature
        )

        return log_loss(
            labels,
            calibrated,
            labels=np.arange(NUM_CLASSES)
        )

    result = minimize_scalar(
        objective,
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-5}
    )

    return float(result.x), float(result.fun)


# =========================================================
# EXPECTED CALIBRATION ERROR
# =========================================================

def calculate_ece(probabilities, labels, bins=15):

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)

    correct = (predictions == labels).astype(float)

    boundaries = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0

    for i in range(bins):

        if i == bins - 1:
            mask = (
                (confidence >= boundaries[i]) &
                (confidence <= boundaries[i + 1])
            )
        else:
            mask = (
                (confidence >= boundaries[i]) &
                (confidence < boundaries[i + 1])
            )

        if np.any(mask):

            accuracy = correct[mask].mean()
            avg_confidence = confidence[mask].mean()

            ece += (
                mask.mean()
                * abs(accuracy - avg_confidence)
            )

    return float(ece)


# =========================================================
# CALIBRATE CNN AND SIFT SEPARATELY
# =========================================================

print("\n" + "=" * 65)
print("FITTING TEMPERATURES ON VALIDATION SET")
print("=" * 65)

cnn_temperature, cnn_nll = fit_temperature(
    cnn_val,
    y_val
)

sift_temperature, sift_nll = fit_temperature(
    sift_val,
    y_val
)

print(f"CNN temperature: {cnn_temperature:.5f}")
print(f"SIFT temperature: {sift_temperature:.5f}")


# =========================================================
# CALIBRATED PROBABILITIES
# =========================================================

cnn_val_cal = apply_temperature(
    cnn_val,
    cnn_temperature
)

sift_val_cal = apply_temperature(
    sift_val,
    sift_temperature
)

cnn_test_cal = apply_temperature(
    cnn_test,
    cnn_temperature
)

sift_test_cal = apply_temperature(
    sift_test,
    sift_temperature
)


# =========================================================
# CALIBRATION COMPARISON
# =========================================================

calibration_rows = []

for name, raw_val, calibrated_val in [
    ("CNN", cnn_val, cnn_val_cal),
    ("SIFT", sift_val, sift_val_cal)
]:

    calibration_rows.append({
        "model": name,
        "raw_nll": log_loss(
            y_val,
            raw_val,
            labels=np.arange(NUM_CLASSES)
        ),
        "calibrated_nll": log_loss(
            y_val,
            calibrated_val,
            labels=np.arange(NUM_CLASSES)
        ),
        "raw_ece": calculate_ece(raw_val, y_val),
        "calibrated_ece": calculate_ece(
            calibrated_val, y_val
        ),
        "temperature": (
            cnn_temperature if name == "CNN"
            else sift_temperature
        )
    })

calibration_df = pd.DataFrame(calibration_rows)

calibration_df.to_csv(
    OUT / "calibration_comparison.csv",
    index=False
)

print("\nCalibration comparison:")
print(calibration_df.to_string(index=False))


# =========================================================
# RAW VS CALIBRATED FUSION
# =========================================================

print("\n" + "=" * 65)
print("VALIDATION ALPHA SEARCH")
print("=" * 65)

alpha_results = []

for mode, cnn_probs, sift_probs in [
    ("raw", cnn_val, sift_val),
    ("temperature_scaled", cnn_val_cal, sift_val_cal)
]:

    for alpha in ALPHAS:

        fused = (
            alpha * cnn_probs
            + (1.0 - alpha) * sift_probs
        )

        pred = fused.argmax(axis=1)

        alpha_results.append({
            "mode": mode,
            "alpha_cnn": float(alpha),
            "alpha_sift": float(1.0 - alpha),
            "val_accuracy": accuracy_score(y_val, pred),
            "val_macro_f1": f1_score(
                y_val,
                pred,
                average="macro",
                zero_division=0
            ),
            "val_weighted_f1": f1_score(
                y_val,
                pred,
                average="weighted",
                zero_division=0
            ),
            "val_nll": log_loss(
                y_val,
                fused,
                labels=np.arange(NUM_CLASSES)
            )
        })

alpha_df = pd.DataFrame(alpha_results)

alpha_df.to_csv(
    OUT / "validation_alpha_comparison.csv",
    index=False
)

# Select calibrated alpha using validation Macro-F1.
scaled_df = alpha_df[
    alpha_df["mode"] == "temperature_scaled"
].copy()

scaled_df = scaled_df.sort_values(
    ["val_macro_f1", "val_accuracy", "alpha_cnn"],
    ascending=[False, False, False]
).reset_index(drop=True)

best = scaled_df.iloc[0]

best_alpha = float(best["alpha_cnn"])

print("\nBest calibrated alpha:", best_alpha)
print("Best validation Macro-F1:", best["val_macro_f1"])


# =========================================================
# TEST EVALUATION
# =========================================================

# Select alpha using validation only.
# Test is evaluated after selection.

fused_test_raw = (
    best_alpha * cnn_test
    + (1.0 - best_alpha) * sift_test
)

fused_test_scaled = (
    best_alpha * cnn_test_cal
    + (1.0 - best_alpha) * sift_test_cal
)


def evaluate(name, probabilities):

    predictions = probabilities.argmax(axis=1)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "nll": log_loss(
            y_test,
            probabilities,
            labels=np.arange(NUM_CLASSES)
        ),
        "ece": calculate_ece(
            probabilities,
            y_test
        )
    }


test_results = [
    evaluate("CNN_raw", cnn_test),
    evaluate("SIFT_raw", sift_test),
    evaluate("CNN_temperature_scaled", cnn_test_cal),
    evaluate("SIFT_temperature_scaled", sift_test_cal),
    evaluate("Late_fusion_raw", fused_test_raw),
    evaluate("Late_fusion_temperature_scaled", fused_test_scaled)
]

test_df = pd.DataFrame(test_results)

test_df.to_csv(
    OUT / "final_test_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

print(test_df.to_string(index=False))


# =========================================================
# SAVE FINAL OUTPUTS
# =========================================================

final_predictions = fused_test_scaled.argmax(axis=1)

np.save(
    OUT / "temperature_scaled_late_fusion_test_probabilities.npy",
    fused_test_scaled
)

pd.DataFrame({
    "true_label": y_test,
    "predicted_label": final_predictions
}).to_csv(
    OUT / "temperature_scaled_late_fusion_predictions.csv",
    index=False
)

with open(OUT / "final_results.json", "w") as f:

    json.dump({
        "cnn_temperature": cnn_temperature,
        "sift_temperature": sift_temperature,
        "selected_alpha_cnn": best_alpha,
        "selected_alpha_sift": 1.0 - best_alpha,
        "selection_metric": "validation_macro_f1",
        "calibration_method": "temperature_scaling",
        "test_results": test_results
    }, f, indent=4)

print("\nExperiment completed successfully.")
print("Results saved in:", OUT)

