import cv2
import numpy as np
import pickle
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "sift"

VOCAB_SIZE = 256
MAX_DESCRIPTORS = 100_000
RANDOM_SEED = 42

SVM_C = 10


# ============================================================
# SETUP
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sift = cv2.SIFT_create()

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# GET DATA
# ============================================================

def get_images(split):
    split_dir = DATA_ROOT / split

    samples = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.glob("*.jpg")):
            samples.append((image_path, class_dir.name))

    return samples


train_samples = get_images("train")
val_samples = get_images("val")
test_samples = get_images("test")

classes = sorted(set(label for _, label in train_samples))
class_to_id = {name: i for i, name in enumerate(classes)}

print("=" * 60)
print("DATASET")
print("=" * 60)
print(f"Classes       : {len(classes)}")
print(f"Train images  : {len(train_samples)}")
print(f"Val images    : {len(val_samples)}")
print(f"Test images   : {len(test_samples)}")


# ============================================================
# SIFT EXTRACTION
# ============================================================

def extract_descriptors(image_path):

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    _, descriptors = sift.detectAndCompute(image, None)

    return descriptors


# ============================================================
# BUILD VISUAL VOCABULARY
# ============================================================

print("\n" + "=" * 60)
print("BUILDING SIFT VISUAL VOCABULARY")
print("=" * 60)

all_descriptors = []

for i, (image_path, _) in enumerate(train_samples):

    descriptors = extract_descriptors(image_path)

    if descriptors is not None:
        all_descriptors.append(descriptors)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(train_samples)} training images")


all_descriptors = np.vstack(all_descriptors)

print(f"\nTotal descriptors collected: {len(all_descriptors):,}")


# Randomly sample descriptors for K-Means
if len(all_descriptors) > MAX_DESCRIPTORS:

    indices = rng.choice(
        len(all_descriptors),
        size=MAX_DESCRIPTORS,
        replace=False
    )

    vocabulary_descriptors = all_descriptors[indices]

else:
    vocabulary_descriptors = all_descriptors


print(
    f"Descriptors used for vocabulary: "
    f"{len(vocabulary_descriptors):,}"
)


# Free memory
del all_descriptors


# ============================================================
# K-MEANS
# ============================================================

print("\nTraining MiniBatch K-Means...")

kmeans = MiniBatchKMeans(
    n_clusters=VOCAB_SIZE,
    random_state=RANDOM_SEED,
    batch_size=2048,
    n_init=3,
    verbose=1
)

kmeans.fit(vocabulary_descriptors)

print("Vocabulary training complete.")


# Save vocabulary
with open(RESULTS_DIR / "sift_vocabulary.pkl", "wb") as f:
    pickle.dump(kmeans, f)


# ============================================================
# CONVERT IMAGE → BOVW HISTOGRAM
# ============================================================

def image_to_bovw(image_path):

    descriptors = extract_descriptors(image_path)

    histogram = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if descriptors is None:
        return histogram

    visual_words = kmeans.predict(descriptors)

    for word in visual_words:
        histogram[word] += 1

    # L1 normalization
    total = histogram.sum()

    if total > 0:
        histogram /= total

    return histogram


def build_features(samples, split_name):

    X = []
    y = []

    print(f"\nBuilding BoVW features: {split_name}")

    for i, (image_path, label) in enumerate(samples):

        feature = image_to_bovw(image_path)

        X.append(feature)
        y.append(class_to_id[label])

        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(samples)}")

    return np.asarray(X), np.asarray(y)


# ============================================================
# BUILD FEATURES
# ============================================================

X_train, y_train = build_features(train_samples, "train")
X_val, y_val = build_features(val_samples, "validation")
X_test, y_test = build_features(test_samples, "test")

print("\nFeature shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# ============================================================
# SVM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SVM")
print("=" * 60)

svm = SVC(
    C=SVM_C,
    kernel="rbf",
    probability=False,
    random_state=RANDOM_SEED
)

svm.fit(X_train, y_train)

print("SVM training complete.")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

val_predictions = svm.predict(X_val)

val_accuracy = accuracy_score(y_val, val_predictions)
val_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Validation Macro-F1 : {val_f1:.4f}")


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST")
print("=" * 60)

test_predictions = svm.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")


# ============================================================
# REPORT
# ============================================================

report = classification_report(
    y_test,
    test_predictions,
    target_names=classes
)

print("\nClassification Report:")
print(report)


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(RESULTS_DIR / "X_test.npy", X_test)
np.save(RESULTS_DIR / "y_test.npy", y_test)
np.save(RESULTS_DIR / "test_predictions.npy", test_predictions)

with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

with open(RESULTS_DIR / "metrics.txt", "w") as f:
    f.write(f"Vocabulary size: {VOCAB_SIZE}\n")
    f.write(f"SVM C: {SVM_C}\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
    f.write(f"Train images: {len(train_samples)}\n")
    f.write(f"Validation images: {len(val_samples)}\n")
    f.write(f"Test images: {len(test_samples)}\n")
    f.write(f"Validation Accuracy: {val_accuracy:.6f}\n")
    f.write(f"Validation Macro-F1: {val_f1:.6f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.6f}\n")
    f.write(f"Test Macro-F1: {test_f1:.6f}\n")

with open(RESULTS_DIR / "svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("\n" + "=" * 60)
print("SIFT BASELINE COMPLETE")
print("=" * 60)
print(f"Results saved to: {RESULTS_DIR}")
