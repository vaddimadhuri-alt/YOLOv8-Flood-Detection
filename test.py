from ultralytics import YOLO
from pathlib import Path

# 1. Load trained model
model_path = Path("weights/best.pt")

if not model_path.exists():
    raise FileNotFoundError("weights/best.pt not found")

model = YOLO(str(model_path))

print("Trained YOLOv8 model loaded successfully!")

# 2. Test dataset folder
test_folder = Path("dataset/images/test")

if not test_folder.exists():
    raise FileNotFoundError("Test images folder not found")

# 3. Run object detection
results = model.predict(
    source=str(test_folder),
    conf=0.25,
    save=True,
    project="runs",
    name="test_predictions",
    exist_ok=True
)

# 4. Display results
print("\n========== TEST RESULTS ==========")

for result in results:
    print("Image:", Path(result.path).name)
    print("Detected objects:", len(result.boxes))

print("\nTesting completed successfully!")
print("Check: runs/detect/test_predictions/")
