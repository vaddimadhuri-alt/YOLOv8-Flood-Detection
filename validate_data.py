import cv2
import pandas as pd
from pathlib import Path

BASE = Path.home() / "Desktop" / "OD"

IMAGE_DIR = BASE / "Image"
MASK_DIR = BASE / "Mask"
CSV_FILE = BASE / "metadata.csv"

df = pd.read_csv(CSV_FILE)

valid = 0
missing = 0
corrupted = 0
dimension_errors = 0
empty_masks = 0

for _, row in df.iterrows():

    image_path = IMAGE_DIR / str(row["Image"])
    mask_path = MASK_DIR / str(row["Mask"])

    if not image_path.exists() or not mask_path.exists():
        missing += 1
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), 0)

    if image is None or mask is None:
        corrupted += 1
        continue

    if image.shape[:2] != mask.shape[:2]:
        dimension_errors += 1
        continue

    if cv2.countNonZero(mask) == 0:
        empty_masks += 1

    valid += 1

print("\nDATA VALIDATION REPORT")
print("----------------------")
print("Total records:", len(df))
print("Valid pairs:", valid)
print("Missing pairs:", missing)
print("Corrupted pairs:", corrupted)
print("Dimension errors:", dimension_errors)
print("Empty masks:", empty_masks)

print("\nValidation completed!")

