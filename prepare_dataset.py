import cv2
import pandas as pd
import random
import shutil
from pathlib import Path

# ======================================
# 1. PATHS
# ======================================

PROJECT = Path(__file__).resolve().parent
RAW = Path.home() / "Desktop" / "OD"

IMAGE_DIR = RAW / "Image"
MASK_DIR = RAW / "Mask"
CSV_FILE = RAW / "metadata.csv"

OUTPUT = PROJECT / "dataset"
PREVIEW = PROJECT / "original_images_with_boxes"

MIN_AREA = 30
RANDOM_SEED = 42

# ======================================
# 2. VALIDATE PATHS
# ======================================

for path in [IMAGE_DIR, MASK_DIR, CSV_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

df = pd.read_csv(CSV_FILE)

if not {"Image", "Mask"}.issubset(df.columns):
    raise ValueError("CSV needs Image and Mask columns")

print("Total records:", len(df))

# ======================================
# 3. CLEAN OLD OUTPUT
# ======================================

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

if PREVIEW.exists():
    shutil.rmtree(PREVIEW)

# ======================================
# 4. READ IMAGES AND GENERATE LABELS
# ======================================

samples = []

valid_pairs = 0
missing_pairs = 0
dimension_errors = 0
empty_masks = 0
total_boxes = 0

for _, row in df.iterrows():

    image_path = IMAGE_DIR / str(row["Image"])
    mask_path = MASK_DIR / str(row["Mask"])

    if not image_path.exists() or not mask_path.exists():
        missing_pairs += 1
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        missing_pairs += 1
        continue

    height, width = image.shape[:2]

    if mask.shape[:2] != (height, width):
        dimension_errors += 1
        continue

    # Convert grayscale mask into binary mask
    _, binary = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find connected flood regions
    count, component_map, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    annotations = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        # YOLO normalized coordinates
        xc = (x + w / 2) / width
        yc = (y + h / 2) / height

        nw = w / width
        nh = h / height

        annotations.append(
            f"0 {xc:.6f} {yc:.6f} "
            f"{nw:.6f} {nh:.6f}"
        )

    if len(annotations) == 0:
        empty_masks += 1

    total_boxes += len(annotations)

    samples.append({
        "image_path": image_path,
        "image_name": str(row["Image"]),
        "annotations": annotations
    })

    valid_pairs += 1

print("\nDATA PREPARATION")
print("Valid image pairs:", valid_pairs)
print("Missing/corrupted pairs:", missing_pairs)
print("Dimension errors:", dimension_errors)
print("Images without boxes:", empty_masks)
print("Total bounding boxes:", total_boxes)

if valid_pairs < 10:
    raise RuntimeError("Insufficient valid image pairs")

# ======================================
# 5. SPLIT ALL VALID IMAGES
# ======================================

random.seed(RANDOM_SEED)
random.shuffle(samples)

n = len(samples)

train_end = int(n * 0.70)
val_end = int(n * 0.90)

splits = {
    "train": samples[:train_end],
    "val": samples[train_end:val_end],
    "test": samples[val_end:]
}

# ======================================
# 6. CREATE YOLO FOLDERS
# ======================================

for split in ["train", "val", "test"]:

    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

PREVIEW.mkdir(parents=True, exist_ok=True)

# ======================================
# 7. SAVE ORIGINAL IMAGES AND LABELS
# ======================================

preview_count = 0

for split_name, items in splits.items():

    for sample in items:

        image_path = sample["image_path"]
        image_name = sample["image_name"]

        image = cv2.imread(str(image_path))

        stem = Path(image_name).stem

        # Save clean original image
        shutil.copy2(
            image_path,
            OUTPUT / "images" / split_name / image_name
        )

        # Save YOLO annotation
        label_path = (
            OUTPUT / "labels" /
            split_name / f"{stem}.txt"
        )

        label_path.write_text(
            "\n".join(sample["annotations"]),
            encoding="utf-8"
        )

        # Generate previews for only 10 images
        if preview_count < 10:

            preview = image.copy()

            h, w = image.shape[:2]

            for line in sample["annotations"]:

                values = list(map(float, line.split()))

                _, xc, yc, nw, nh = values

                x1 = int((xc - nw / 2) * w)
                y1 = int((yc - nh / 2) * h)
                x2 = int((xc + nw / 2) * w)
                y2 = int((yc + nh / 2) * h)

                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    preview,
                    "Flooded Area",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            cv2.imwrite(
                str(PREVIEW / f"{stem}_boxed.jpg"),
                preview
            )

            preview_count += 1

# ======================================
# 8. CREATE data.yaml
# ======================================

yaml_text = f"""path: '{OUTPUT.as_posix()}'
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: Flooded_Area
"""

(OUTPUT / "data.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)

# ======================================
# 9. FINAL REPORT
# ======================================

print("\n========== FINAL REPORT ==========")

for split_name, items in splits.items():
    print(split_name, "images:", len(items))

print("Total images:", n)
print("Total boxes:", total_boxes)
print("Preview images:", preview_count)
print("Dataset folder:", OUTPUT)
print("Preview folder:", PREVIEW)

print("\nDATASET PREPARATION COMPLETED!")
