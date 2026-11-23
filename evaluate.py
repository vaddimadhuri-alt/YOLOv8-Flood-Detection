from ultralytics import YOLO
from pathlib import Path

# Load trained model
model = YOLO("weights/best.pt")

# Check dataset configuration
data_yaml = Path("dataset/data.yaml")

if not data_yaml.exists():
    raise FileNotFoundError("data.yaml not found")

# Evaluate using test dataset
metrics = model.val(
    data=str(data_yaml),
    split="test",
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    plots=True
)

# Display evaluation metrics
print("\n========== MODEL EVALUATION ==========")

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

print("\nEvaluation completed!")
