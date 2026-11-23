import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="YOLOv8 Flood Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 YOLOv8 Flood Detection System")
st.write("AI-powered flood detection using computer vision.")

@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

try:
    model = load_model()
    st.success("YOLOv8 model loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an original flood image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Flood Objects"):
        with st.spinner("Detecting objects..."):
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp:
                    image.save(temp.name)
                    temp_path = temp.name

                results = model.predict(
                    source=temp_path,
                    conf=0.25
                )

                result = results[0]
                annotated = result.plot()

                st.subheader("Detection Results")
                st.image(
                    annotated[:, :, ::-1],
                    caption="Detected objects with bounding boxes",
                    use_container_width=True
                )

                st.metric(
                    "Total Detections",
                    len(result.boxes)
                )

                if len(result.boxes) > 0:
                    st.subheader("Detection Details")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        st.write(
                            f"Class: {model.names[class_id]} | "
                            f"Confidence: {confidence:.2%}"
                        )
                else:
                    st.info("No objects detected.")

            except Exception as e:
                st.error(f"Detection failed: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
