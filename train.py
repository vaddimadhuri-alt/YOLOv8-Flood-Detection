import shutil
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = PROJECT_DIR / "dataset" / "data.yaml"

RUNS_DIR = PROJECT_DIR / "runs"

WEIGHTS_DIR = PROJECT_DIR / "weights"

WEIGHTS_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. CHECK DATASET
# ==========================================

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset configuration not found: {DATA_YAML}"
    )

print("Dataset found successfully!")
print("Dataset configuration:", DATA_YAML)

# ==========================================
# 3. LOAD YOLOv8 NANO MODEL
# ==========================================

print("\nLoading YOLOv8 Nano pretrained model...")

model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully!")

# ==========================================
# 4. TRAIN MODEL USING CPU
# ==========================================

print("\nStarting YOLOv8 training on CPU...")

results = model.train(

    data=str(DATA_YAML),

    epochs=20,

    imgsz=416,

    batch=1,

    workers=0,

    device="cpu",

    cache=False,

    project=str(RUNS_DIR),

    name="flood_detection",

    patience=5,

    plots=True,

    exist_ok=True,

    verbose=True
)

print("\nTraining completed!")

# ==========================================
# 5. FIND BEST MODEL
# ==========================================

save_dir = Path(model.trainer.save_dir)

best_model_path = save_dir / "weights" / "best.pt"

print("Training output directory:", save_dir)

if not best_model_path.exists():

    raise FileNotFoundError(
        f"Best model not found at: {best_model_path}"
    )

# ==========================================
# 6. SAVE FINAL BEST.PT
# ==========================================

destination = WEIGHTS_DIR / "best.pt"

shutil.copy2(
    best_model_path,
    destination
)

print("\n================================")
print("MODEL TRAINING SUCCESSFUL!")
print("================================")

print("Best model:", destination)
print("Model size:", round(destination.stat().st_size / (1024 * 1024), 2), "MB")

print("\nYour best.pt file is ready for deployment!")
