import os
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED = 42
RESULTS_DIR = "results/hybrid"


def load(name):
    return np.load(os.path.join(RESULTS_DIR, name))


print("Loading features...")

X_sift_train = load("sift_train.npy")
X_sift_val = load("sift_val.npy")
X_sift_test = load("sift_test.npy")

X_cnn_train = load("cnn_train.npy")
X_cnn_val = load("cnn_val.npy")
X_cnn_test = load("cnn_test.npy")

y_train = load("y_train.npy")
y_val = load("y_val.npy")
y_test = load("y_test.npy")

# Verify labels agree across the feature extraction pipelines
assert np.array_equal(y_train, load("cnn_y_train.npy"))
assert np.array_equal(y_val, load("cnn_y_val.npy"))
assert np.array_equal(y_test, load("cnn_y_test.npy"))

print("Label alignment verified.")


# ---------------------------------------------------------
# Standardize each representation independently
# ---------------------------------------------------------

sift_scaler = StandardScaler()
cnn_scaler = StandardScaler()

X_sift_train_s = sift_scaler.fit_transform(X_sift_train)
X_sift_val_s = sift_scaler.transform(X_sift_val)
X_sift_test_s = sift_scaler.transform(X_sift_test)

X_cnn_train_s = cnn_scaler.fit_transform(X_cnn_train)
X_cnn_val_s = cnn_scaler.transform(X_cnn_val)
X_cnn_test_s = cnn_scaler.transform(X_cnn_test)


# ---------------------------------------------------------
# Hybrid representation
# ---------------------------------------------------------

X_hybrid_train = np.concatenate(
    [X_cnn_train_s, X_sift_train_s], axis=1
)

X_hybrid_val = np.concatenate(
    [X_cnn_val_s, X_sift_val_s], axis=1
)

X_hybrid_test = np.concatenate(
    [X_cnn_test_s, X_sift_test_s], axis=1
)


# ---------------------------------------------------------
# Same classifier for all three
# ---------------------------------------------------------

def train_and_predict(name, X_train, X_val, X_test):

    print(f"\nTraining {name}...")

    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=SEED
    )

    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"{name}")
    print(f"  Val Accuracy : {val_acc * 100:.2f}%")
    print(f"  Val Macro-F1 : {val_f1 * 100:.2f}%")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Test Macro-F1: {test_f1 * 100:.2f}%")

    np.save(
        os.path.join(RESULTS_DIR, f"{name.lower()}_predictions.npy"),
        test_pred
    )

    return test_pred


sift_pred = train_and_predict(
    "SIFT",
    X_sift_train_s,
    X_sift_val_s,
    X_sift_test_s
)

cnn_pred = train_and_predict(
    "CNN",
    X_cnn_train_s,
    X_cnn_val_s,
    X_cnn_test_s
)

hybrid_pred = train_and_predict(
    "Hybrid",
    X_hybrid_train,
    X_hybrid_val,
    X_hybrid_test
)


# ---------------------------------------------------------
# Error overlap
# ---------------------------------------------------------

sift_correct = sift_pred == y_test
cnn_correct = cnn_pred == y_test
hybrid_correct = hybrid_pred == y_test

both_correct = cnn_correct & sift_correct
cnn_only = cnn_correct & ~sift_correct
sift_only = ~cnn_correct & sift_correct
both_wrong = ~cnn_correct & ~sift_correct

n = len(y_test)

print("\n" + "=" * 60)
print("CNN–SIFT ERROR OVERLAP")
print("=" * 60)

groups = {
    "Both correct": both_correct,
    "CNN correct / SIFT wrong": cnn_only,
    "CNN wrong / SIFT correct": sift_only,
    "Both wrong": both_wrong,
}

for name, mask in groups.items():
    count = mask.sum()
    print(
        f"{name:35s}: "
        f"{count:4d} / {n} "
        f"({count / n * 100:.2f}%)"
    )


# ---------------------------------------------------------
# Rescue rates
# ---------------------------------------------------------

cnn_errors = ~cnn_correct
sift_errors = ~sift_correct

sift_rescue = (sift_correct & cnn_errors).sum()
cnn_rescue = (cnn_correct & sift_errors).sum()

print("\n" + "=" * 60)
print("RESCUE ANALYSIS")
print("=" * 60)

print(
    f"SIFT rescues CNN errors: "
    f"{sift_rescue} / {cnn_errors.sum()} "
    f"= {sift_rescue / cnn_errors.sum() * 100:.2f}%"
)

print(
    f"CNN rescues SIFT errors: "
    f"{cnn_rescue} / {sift_errors.sum()} "
    f"= {cnn_rescue / sift_errors.sum() * 100:.2f}%"
)


# ---------------------------------------------------------
# Hybrid recovery analysis
# ---------------------------------------------------------

hybrid_correct_mask = hybrid_correct

cnn_wrong_sift_right = (~cnn_correct) & sift_correct
cnn_wrong_sift_wrong = (~cnn_correct) & (~sift_correct)

print("\n" + "=" * 60)
print("HYBRID RECOVERY")
print("=" * 60)

print(
    "CNN wrong + SIFT correct + Hybrid correct:",
    (
        (~cnn_correct)
        & sift_correct
        & hybrid_correct
    ).sum()
)

print(
    "CNN wrong + SIFT correct + Hybrid wrong:",
    (
        (~cnn_correct)
        & sift_correct
        & (~hybrid_correct)
    ).sum()
)

print(
    "CNN correct + SIFT wrong + Hybrid wrong:",
    (
        cnn_correct
        & (~sift_correct)
        & (~hybrid_correct)
    ).sum()
)

print(
    "CNN correct + SIFT wrong + Hybrid correct:",
    (
        cnn_correct
        & (~sift_correct)
        & hybrid_correct
    ).sum()
)


# ---------------------------------------------------------
# Save detailed per-image analysis
# ---------------------------------------------------------

analysis = []

for i in range(n):
    analysis.append({
        "index": int(i),
        "true_label": int(y_test[i]),
        "sift_prediction": int(sift_pred[i]),
        "cnn_prediction": int(cnn_pred[i]),
        "hybrid_prediction": int(hybrid_pred[i]),
        "sift_correct": bool(sift_correct[i]),
        "cnn_correct": bool(cnn_correct[i]),
        "hybrid_correct": bool(hybrid_correct[i]),
        "category": (
            "both_correct"
            if both_correct[i]
            else "cnn_correct_sift_wrong"
            if cnn_only[i]
            else "cnn_wrong_sift_correct"
            if sift_only[i]
            else "both_wrong"
        )
    })

with open(
    os.path.join(RESULTS_DIR, "complementarity_analysis.json"),
    "w"
) as f:
    json.dump(analysis, f, indent=2)


# ---------------------------------------------------------
# Classification reports
# ---------------------------------------------------------

for name, pred in [
    ("sift", sift_pred),
    ("cnn", cnn_pred),
    ("hybrid", hybrid_pred)
]:
    report = classification_report(
        y_test,
        pred,
        zero_division=0
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            f"{name}_controlled_classification_report.txt"
        ),
        "w"
    ) as f:
        f.write(report)


print("\nAnalysis complete.")
print(f"Saved results to: {RESULTS_DIR}")
