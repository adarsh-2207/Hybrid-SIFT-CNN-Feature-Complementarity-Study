
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
C_VALUES = [0.01, 0.1, 1, 10]
ALPHAS = np.round(np.arange(0, 1.001, 0.05), 2)


def metrics(y_true, y_pred, classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=classes,
                average="macro",
                zero_division=0
            )
        )
    }


def run_experiment(name, cnn_dir, sift_dir, output_dir):
    print(f"\n{'='*55}\n{name}\n{'='*55}")

    # Load CNN features and labels
    cnn_dir = ROOT / cnn_dir
    sift_dir = ROOT / sift_dir
    output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    Xc_train = np.load(cnn_dir / "cnn_train.npy")
    Xc_val = np.load(cnn_dir / "cnn_val.npy")
    Xc_test = np.load(cnn_dir / "cnn_test.npy")

    yc_train = np.load(cnn_dir / "cnn_y_train.npy")
    yc_val = np.load(cnn_dir / "cnn_y_val.npy")
    yc_test = np.load(cnn_dir / "cnn_y_test.npy")

    # Load SIFT features and labels
    if name == "Caltech-101":
        Xs_train = np.load(sift_dir / "sift_train.npy")
        Xs_val = np.load(sift_dir / "sift_val.npy")
        Xs_test = np.load(sift_dir / "sift_test.npy")
    else:
        Xs_train = np.load(sift_dir / "X_train.npy")
        Xs_val = np.load(sift_dir / "X_val.npy")
        Xs_test = np.load(sift_dir / "X_test.npy")

    ys_train = np.load(sift_dir / "y_train.npy")
    ys_val = np.load(sift_dir / "y_val.npy")
    ys_test = np.load(sift_dir / "y_test.npy")

    # Verify matching samples and label order
    assert np.array_equal(yc_train, ys_train), "Train labels mismatch"
    assert np.array_equal(yc_val, ys_val), "Validation labels mismatch"
    assert np.array_equal(yc_test, ys_test), "Test labels mismatch"

    y_train, y_val, y_test = yc_train, yc_val, yc_test
    classes = np.unique(y_train)

    print("Feature shapes:", Xc_train.shape, Xs_train.shape)
    print("Train/Val/Test:", len(y_train), len(y_val), len(y_test))

    # Scale each feature type independently
    scaler_c = StandardScaler()
    scaler_s = StandardScaler()

    Xc_train = scaler_c.fit_transform(Xc_train)
    Xc_val = scaler_c.transform(Xc_val)
    Xc_test = scaler_c.transform(Xc_test)

    Xs_train = scaler_s.fit_transform(Xs_train)
    Xs_val = scaler_s.transform(Xs_val)
    Xs_test = scaler_s.transform(Xs_test)

    # Select C independently for CNN and SIFT using validation Macro-F1
    models = {}
    probabilities = {}

    for modality, X_train, X_val, X_test in [
        ("CNN", Xc_train, Xc_val, Xc_test),
        ("SIFT", Xs_train, Xs_val, Xs_test)
    ]:
        best_model = None
        best_c = None
        best_score = -1

        for c in C_VALUES:
            model = LogisticRegression(
                C=c,
                max_iter=3000,
                solver="lbfgs"
            )
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            score = metrics(y_val, val_pred, classes)["macro_f1"]

            print(f"{modality} C={c}: validation Macro-F1={score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_c = c

        models[modality] = best_model

        assert np.array_equal(best_model.classes_, classes)

        probabilities[modality] = {
            "val": best_model.predict_proba(X_val),
            "test": best_model.predict_proba(X_test)
        }

        print(f"Selected {modality} C={best_c}")

    # Select alpha using validation only
    # P_final = alpha * P_CNN + (1-alpha) * P_SIFT
    rows = []

    for alpha in ALPHAS:
        p_val = (
            alpha * probabilities["CNN"]["val"]
            + (1 - alpha) * probabilities["SIFT"]["val"]
        )

        pred_val = classes[np.argmax(p_val, axis=1)]
        result = metrics(y_val, pred_val, classes)

        rows.append({
            "alpha_cnn": float(alpha),
            "alpha_sift": float(1 - alpha),
            **result
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "validation_alpha_results.csv", index=False)

    # Highest Macro-F1; accuracy breaks ties
    results_df = results_df.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False
    )

    best = results_df.iloc[0]
    best_alpha = float(best["alpha_cnn"])

    print("\nBest validation fusion weight:")
    print(best.to_dict())

    # Final test evaluation using selected alpha
    p_test = (
        best_alpha * probabilities["CNN"]["test"]
        + (1 - best_alpha) * probabilities["SIFT"]["test"]
    )

    hybrid_pred = classes[np.argmax(p_test, axis=1)]

    cnn_pred = classes[
        np.argmax(probabilities["CNN"]["test"], axis=1)
    ]
    sift_pred = classes[
        np.argmax(probabilities["SIFT"]["test"], axis=1)
    ]

    final = {
        "dataset": name,
        "selected_alpha_cnn": best_alpha,
        "selected_alpha_sift": 1 - best_alpha,
        "validation_macro_f1": float(best["macro_f1"]),
        "test_cnn_only": metrics(y_test, cnn_pred, classes),
        "test_sift_only": metrics(y_test, sift_pred, classes),
        "test_late_fusion": metrics(y_test, hybrid_pred, classes)
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final, f, indent=4)

    np.save(output_dir / "late_fusion_test_predictions.npy", hybrid_pred)
    np.save(output_dir / "cnn_only_test_predictions.npy", cnn_pred)
    np.save(output_dir / "late_fusion_test_probabilities.npy", p_test)

    print("\nFINAL TEST RESULTS")
    print(json.dumps(final, indent=4))
    print(f"Saved in: {output_dir}")


if __name__ == "__main__":
    run_experiment(
        "Caltech-101",
        "results/hybrid",
        "results/hybrid",
        "results/late_fusion_caltech101"
    )

    run_experiment(
        "Caltech-256",
        "results_caltech256/hybrid_features",
        "results_caltech256/sift_bovw_svm",
        "results_caltech256/late_fusion"
    )

