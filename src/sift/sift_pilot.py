import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "train"

sift = cv2.SIFT_create()

count = 0

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg"))[:5]

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read: {image_path}")
            continue

        keypoints, descriptors = sift.detectAndCompute(image, None)

        print(
            f"{class_dir.name:20s} "
            f"{image_path.name:15s} "
            f"keypoints={len(keypoints):4d} "
            f"descriptor_shape={descriptors.shape if descriptors is not None else None}"
        )

        count += 1

print(f"\nProcessed {count} images.")
