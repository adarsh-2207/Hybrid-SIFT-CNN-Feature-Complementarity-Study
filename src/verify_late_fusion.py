
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_caltech256" / "late_fusion"

# Load predictions
y_true = np.load(ROOT / "results_caltech256" / "sift_bovw_svm" / "y_test.npy")
cnn_pred = np.load(OUT / "cnn_only_test_predictions.npy") if (OUT / "cnn_only_test_predictions.npy").exists() else None
fusion_pred = np.load(OUT / "late_fusion_test_predictions.npy")

# Recreate CNN predictions from saved test probabilities if needed
if cnn_pred is None:
    raise FileNotFoundError(
        "CNN-only predictions not found in late_fusion folder. "
        "We need to save them from the experiment first."
    )

assert len(y_true) == len(cnn_pred) == len(fusion_pred)

# McNemar test: compare paired correct/incorrect outcomes
cnn_correct = cnn_pred == y_true
fusion_correct = fusion_pred == y_true

b = int(np.sum(cnn_correct & ~fusion_correct))
c = int(np.sum(~cnn_correct & fusion_correct))

mcnemar = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided")

print("McNemar test")
print("CNN correct, fusion wrong:", b)
print("CNN wrong, fusion correct:", c)
print("Exact two-sided p-value:", mcnemar.pvalue)

# Paired bootstrap for accuracy and Macro-F1 differences
from sklearn.metrics import accuracy_score, f1_score

classes = np.unique(y_true)
rng = np.random.default_rng(42)
n = len(y_true)
B = 10000

accuracy_diffs = []
macro_f1_diffs = []

for _ in range(B):
    idx = rng.integers(0, n, size=n)

    acc_cnn = accuracy_score(y_true[idx], cnn_pred[idx])
    acc_fusion = accuracy_score(y_true[idx], fusion_pred[idx])

    f1_cnn = f1_score(
        y_true[idx], cnn_pred[idx],
        labels=classes, average="macro", zero_division=0
    )
    f1_fusion = f1_score(
        y_true[idx], fusion_pred[idx],
        labels=classes, average="macro", zero_division=0
    )

    accuracy_diffs.append(acc_fusion - acc_cnn)
    macro_f1_diffs.append(f1_fusion - f1_cnn)


def report(name, values):
    values = np.asarray(values)
    low, high = np.percentile(values, [2.5, 97.5])
    print(f"\n{name}")
    print(f"Observed difference: {values.mean() * 100:.4f} pp")
    print(f"95% bootstrap CI: [{low * 100:.4f}, {high * 100:.4f}] pp")


report("Accuracy difference", accuracy_diffs)
report("Macro-F1 difference", macro_f1_diffs)

