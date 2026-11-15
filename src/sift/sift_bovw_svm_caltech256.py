import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG — CALTECH-256
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data_caltech256"
RESULTS_DIR = PROJECT_ROOT / "results_caltech256" / "sift_bovw_svm"

VOCAB_SIZE = 256
MAX_DESCRIPTORS = 100_000
MAX_DESCRIPTORS_PER_IMAGE = 50
RANDOM_SEED = 42
SVM_C = 10

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tif", ".tiff", ".webp",
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)
sift = cv2.SIFT_create()


# ============================================================
# DATA LOADING
# ============================================================

def get_images(split):
    split_dir = DATA_ROOT / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((image_path, class_dir.name))
    return samples


train_samples = get_images("train")
val_samples = get_images("val")
test_samples = get_images("test")

classes = sorted({label for _, label in train_samples})
class_to_id = {name: idx for idx, name in enumerate(classes)}

if len(classes) != 256:
    raise ValueError(f"Expected 256 classes, found {len(classes)}.")

for split_name, samples in [("validation", val_samples), ("test", test_samples)]:
    missing_classes = sorted({label for _, label in samples} - set(class_to_id))
    if missing_classes:
        raise ValueError(f"{split_name} contains unknown classes: {missing_classes[:10]}")

print("=" * 68)
print("CALTECH-256 — SIFT + BoVW + SVM")
print("=" * 68)
print(f"Classes      : {len(classes)}")
print(f"Train images : {len(train_samples)}")
print(f"Val images   : {len(val_samples)}")
print(f"Test images  : {len(test_samples)}")
print(f"Vocabulary   : {VOCAB_SIZE}")
print(f"SVM C        : {SVM_C}")
print(f"Output       : {RESULTS_DIR}")


# ============================================================
# SIFT EXTRACTION
# ============================================================

def extract_descriptors(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"WARNING: Could not read image: {image_path}")
        return None

    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors


# ============================================================
# BUILD VISUAL VOCABULARY USING TRAINING DATA ONLY
# Keep a bounded descriptor sample to control memory use.
# Cap descriptors per image so a few texture-rich images do not
# dominate the candidate pool.
# ============================================================

print("\n" + "=" * 68)
print("COLLECTING TRAINING DESCRIPTORS")
print("=" * 68)

descriptor_chunks = []
images_with_descriptors = 0

for i, (image_path, _) in enumerate(train_samples, start=1):
    descriptors = extract_descriptors(image_path)

    if descriptors is not None and len(descriptors) > 0:
        images_with_descriptors += 1
        if len(descriptors) > MAX_DESCRIPTORS_PER_IMAGE:
            chosen = rng.choice(
                len(descriptors),
                size=MAX_DESCRIPTORS_PER_IMAGE,
                replace=False,
            )
            descriptors = descriptors[chosen]
        descriptor_chunks.append(descriptors.astype(np.float32, copy=False))

    if i % 500 == 0 or i == len(train_samples):
        print(f"Processed {i}/{len(train_samples)} training images")

if not descriptor_chunks:
    raise RuntimeError("No SIFT descriptors were extracted from training images.")

candidate_descriptors = np.vstack(descriptor_chunks)
del descriptor_chunks

print(f"Training images with SIFT descriptors: {images_with_descriptors}")
print(f"Candidate descriptors: {len(candidate_descriptors):,}")

if len(candidate_descriptors) > MAX_DESCRIPTORS:
    chosen = rng.choice(
        len(candidate_descriptors),
        size=MAX_DESCRIPTORS,
        replace=False,
    )
    vocabulary_descriptors = candidate_descriptors[chosen]
else:
    vocabulary_descriptors = candidate_descriptors

del candidate_descriptors
print(f"Descriptors used for vocabulary: {len(vocabulary_descriptors):,}")

if len(vocabulary_descriptors) < VOCAB_SIZE:
    raise RuntimeError(
        f"Only {len(vocabulary_descriptors)} descriptors available; "
        f"need at least {VOCAB_SIZE} for the vocabulary."
    )


# ============================================================
# MINI-BATCH K-MEANS
# ============================================================

print("\nTraining MiniBatch K-Means...")
kmeans = MiniBatchKMeans(
    n_clusters=VOCAB_SIZE,
    random_state=RANDOM_SEED,
    batch_size=2048,
    n_init=3,
    verbose=1,
)
kmeans.fit(vocabulary_descriptors)
del vocabulary_descriptors
print("Vocabulary training complete.")

with open(RESULTS_DIR / "sift_vocabulary.pkl", "wb") as f:
    pickle.dump(kmeans, f)


# ============================================================
# IMAGE → BOVW HISTOGRAM
# ============================================================

def image_to_bovw(image_path):
    histogram = np.zeros(VOCAB_SIZE, dtype=np.float32)
    descriptors = extract_descriptors(image_path)

    if descriptors is None or len(descriptors) == 0:
        return histogram

    visual_words = kmeans.predict(descriptors)
    histogram = np.bincount(visual_words, minlength=VOCAB_SIZE).astype(np.float32)

    total = histogram.sum()
    if total > 0:
        histogram /= total  # L1-normalized histogram

    return histogram


def build_features(samples, split_name):
    X = np.zeros((len(samples), VOCAB_SIZE), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int64)

    print(f"\nBuilding BoVW features: {split_name}")
    for i, (image_path, label) in enumerate(samples, start=1):
        X[i - 1] = image_to_bovw(image_path)
        y[i - 1] = class_to_id[label]

        if i % 500 == 0 or i == len(samples):
            print(f"{split_name}: {i}/{len(samples)}")

    return X, y


X_train, y_train = build_features(train_samples, "train")
X_val, y_val = build_features(val_samples, "validation")
X_test, y_test = build_features(test_samples, "test")

print("\nFeature shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# Save feature matrices for reproducibility and later hybrid experiments.
np.save(RESULTS_DIR / "X_train.npy", X_train)
np.save(RESULTS_DIR / "y_train.npy", y_train)
np.save(RESULTS_DIR / "X_val.npy", X_val)
np.save(RESULTS_DIR / "y_val.npy", y_val)
np.save(RESULTS_DIR / "X_test.npy", X_test)
np.save(RESULTS_DIR / "y_test.npy", y_test)

with open(RESULTS_DIR / "class_to_idx.json", "w") as f:
    json.dump(class_to_id, f, indent=2)


# ============================================================
# SVM — FIXED CONFIG FROM EXISTING PIPELINE
# Validation is reported; test is reserved for final evaluation.
# ============================================================

print("\n" + "=" * 68)
print("TRAINING RBF SVM")
print("=" * 68)

svm = SVC(
    C=SVM_C,
    kernel="rbf",
    probability=False,
    random_state=RANDOM_SEED,
)
svm.fit(X_train, y_train)
print("SVM training complete.")


# ============================================================
# VALIDATION METRICS
# ============================================================

val_predictions = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
val_f1 = f1_score(y_val, val_predictions, average="macro", zero_division=0)

print("\nVALIDATION")
print(f"Validation Accuracy : {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
print(f"Validation Macro-F1 : {val_f1:.4f} ({val_f1 * 100:.2f}%)")


# ============================================================
# FINAL TEST METRICS AND DIAGNOSTICS
# ============================================================

print("\nFINAL TEST")
test_predictions = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=0)

print(f"Test Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Test Macro-F1 : {test_f1:.4f} ({test_f1 * 100:.2f}%)")

report_dict = classification_report(
    y_test,
    test_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

with open(RESULTS_DIR / "per_class_metrics.json", "w") as f:
    json.dump(report_dict, f, indent=2)

with open(RESULTS_DIR / "per_class_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["class", "precision", "recall", "f1-score", "support"]
    )
    writer.writeheader()
    for name in classes:
        row = report_dict[name]
        writer.writerow({
            "class": name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1-score": row["f1-score"],
            "support": row["support"],
        })

cm = confusion_matrix(y_test, test_predictions, labels=list(range(len(classes))))
np.save(RESULTS_DIR / "confusion_matrix.npy", cm)

with open(RESULTS_DIR / "confusion_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true/predicted"] + classes)
    for class_name, row in zip(classes, cm):
        writer.writerow([class_name] + row.tolist())

np.save(RESULTS_DIR / "test_predictions.npy", test_predictions)

with open(RESULTS_DIR / "svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

metrics = {
    "model": "SIFT + BoVW + RBF-SVM",
    "dataset": "Caltech-256",
    "seed": RANDOM_SEED,
    "num_classes": len(classes),
    "vocabulary_size": VOCAB_SIZE,
    "max_descriptors": MAX_DESCRIPTORS,
    "max_descriptors_per_image_for_vocabulary": MAX_DESCRIPTORS_PER_IMAGE,
    "svm_kernel": "rbf",
    "svm_C": SVM_C,
    "train_images": len(train_samples),
    "validation_images": len(val_samples),
    "test_images": len(test_samples),
    "validation_accuracy": val_accuracy,
    "validation_macro_f1": val_f1,
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_f1,
    "vocabulary_fitted_on": "training images only",
    "histogram_normalization": "L1",
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 68)
print("CALTECH-256 SIFT + BoVW + SVM COMPLETE")
print("=" * 68)
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Macro-F1 : {test_f1 * 100:.2f}%")
print(f"Results saved to: {RESULTS_DIR}")
print("Includes metrics, model, vocabulary, feature matrices, confusion matrix,")
print("and per-class precision/recall/F1.")
