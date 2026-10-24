
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Pothole Detection System")
st.write(
    "Upload a road image to detect potholes "
    "using my custom-trained YOLO11n model."
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Potholes", type="primary"):
        with st.spinner("Analyzing image..."):
            result = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )[0]

        # Convert the annotated image from BGR to RGB
        annotated = result.plot()[:, :, ::-1]

        st.subheader("Detection Results")
        st.image(
            annotated,
            caption="Image with detected objects",
            use_container_width=True
        )

        if len(result.boxes) == 0:
            st.info("No objects detected at this confidence level.")
        else:
            rows = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                rows.append({
                    "Detected Class": result.names[class_id],
                    "Confidence": f"{score:.2%}"
                })

            st.subheader("Detected Objects")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"Detected {len(result.boxes)} object(s)."
            )
