import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "hybrid")
VOCAB_PATH = os.path.join(
    PROJECT_ROOT, "results", "sift", "sift_vocabulary.pkl"
)

os.makedirs(RESULTS_ROOT, exist_ok=True)


# ---------------------------------------------------------
# Load frozen SIFT vocabulary
# ---------------------------------------------------------

print("Loading frozen SIFT vocabulary...")

with open(VOCAB_PATH, "rb") as f:
    vocabulary = pickle.load(f)

print("Vocabulary shape:", vocabulary.cluster_centers_.shape)


# ---------------------------------------------------------
# SIFT
# ---------------------------------------------------------

sift = cv2.SIFT_create()


# ---------------------------------------------------------
# Convert one image -> BoVW histogram
# ---------------------------------------------------------

def image_to_bovw(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    histogram = np.zeros(vocabulary.n_clusters, dtype=np.float32)

    if descriptors is not None and len(descriptors) > 0:
        words = vocabulary.predict(descriptors)

        for word in words:
            histogram[word] += 1.0

    # Same L1 normalization used in the SIFT baseline
    total = histogram.sum()

    if total > 0:
        histogram /= total

    return histogram


# ---------------------------------------------------------
# Extract split
# ---------------------------------------------------------

def extract_split(split):
    split_root = os.path.join(DATA_ROOT, split)

    classes = sorted(
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    )

    samples = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    for class_name in classes:
        class_dir = os.path.join(split_root, class_name)

        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(valid_extensions):
                samples.append(
                    (
                        os.path.join(class_dir, filename),
                        class_name
                    )
                )

    print()
    print("=" * 60)
    print(f"Processing {split}")
    print("Classes:", len(classes))
    print("Images:", len(samples))
    print("=" * 60)

    X = np.zeros(
        (len(samples), vocabulary.n_clusters),
        dtype=np.float32
    )

    y = np.zeros(len(samples), dtype=np.int64)

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    for i, (image_path, class_name) in enumerate(
        tqdm(samples, desc=f"SIFT {split}")
    ):
        X[i] = image_to_bovw(image_path)
        y[i] = class_to_idx[class_name]

    np.save(
        os.path.join(RESULTS_ROOT, f"sift_{split}.npy"),
        X
    )

    np.save(
        os.path.join(RESULTS_ROOT, f"y_{split}.npy"),
        y
    )

    print(f"{split} feature shape:", X.shape)
    print(f"{split} label shape:", y.shape)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        extract_split(split)

    print()
    print("=" * 60)
    print("SIFT BoVW feature extraction COMPLETE")
    print("=" * 60)
