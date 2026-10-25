import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("Road Damage Detection System")
st.write(
    "Upload a road video to detect potholes, "
    "cracks, and damaged road areas."
)

# Location of your trained model
MODEL_PATH = Path(__file__).parent / "best.pt"


# Load the model once
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# Check that the model exists
if not MODEL_PATH.exists():
    st.error("best.pt was not found in your GitHub repository.")
    st.stop()


# Upload video
uploaded_video = st.file_uploader(
    "Upload a road video",
    type=["mp4", "mov", "avi"]
)

# Confidence threshold
confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.35,
    step=0.05
)


if uploaded_video is not None:

    # Save uploaded video temporarily
    video_suffix = Path(uploaded_video.name).suffix.lower()

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        input_video = temp_path / f"input{video_suffix}"

        input_video.write_bytes(uploaded_video.getvalue())

        st.subheader("Original Video")
        st.video(uploaded_video.getvalue())

        if st.button("Detect Road Damage", type="primary"):

            output_dir = temp_path / "output"

            try:
                model = load_model()

                with st.spinner(
                    "Processing video... Please wait."
                ):

                    # Run detection frame by frame
                    results = model.predict(
                        source=str(input_video),
                        conf=confidence,
                        imgsz=640,
                        stream=True,
                        save=True,
                        project=str(output_dir),
                        name="results",
                        exist_ok=True,
                        verbose=False
                    )

                    # Process every frame
                    for result in results:
                        pass

                    # Find the generated video
                    result_folder = output_dir / "results"

                    video_files = [
                        path
                        for path in result_folder.iterdir()
                        if path.is_file()
                        and path.suffix.lower() in [
                            ".mp4", ".avi", ".mov", ".mkv"
                        ]
                    ]

                    if not video_files:
                        st.error(
                            "The processed video was not found. "
                            "Check the app logs for errors."
                        )

                    else:
                        processed_video = video_files[0]

                        st.success(
                            "Video processing completed!"
                        )

                        st.subheader("Road Damage Detection Result")

                        st.video(processed_video.read_bytes())

                        st.download_button(
                            label="Download Processed Video",
                            data=processed_video.read_bytes(),
                            file_name="road_damage_detected.mp4",
                            mime="video/mp4"
                        )

            except Exception as error:
                st.error(f"Video processing failed: {error}")
