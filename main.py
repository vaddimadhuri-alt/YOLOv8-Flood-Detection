from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import base64

app = FastAPI(
    title="YOLOv8 Flood Detection API"
)

# Frontend access - restrict to your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load trained model
MODEL_PATH = (
    Path(__file__).resolve().parent /
    "weights" / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError("best.pt not found")

model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return {
        "message": "Flood Detection API is running",
        "model": "YOLOv8"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG or PNG image"
        )

    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        image_array = np.array(image)

        results = model.predict(
            source=image_array,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        detections = []

        for box in result.boxes:

            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(
                    float(box.conf[0]), 4
                ),
                "coordinates": [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ]
            })

        # Draw predicted boxes on original image
        annotated = result.plot()

        success, buffer = cv2.imencode(
            ".jpg", annotated
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Image encoding failed"
            )

        encoded = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        return {
            "total_detections": len(detections),
            "detections": detections,
            "annotated_image": encoded
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
