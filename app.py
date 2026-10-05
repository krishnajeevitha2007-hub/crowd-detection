"""CrowdSense: a real-time crowd monitoring and prediction dashboard."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile

import streamlit as st

try:
    import cv2
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as error:
    st.error(f"A required package is missing: {error.name or error}.")
    st.code("py -m pip install -r requirements.txt")
    st.stop()

from crowd_detection import CrowdDetector, open_video, open_webcam
from crowd_prediction import classify_density, predict_crowd_trend


CSV_PATH = Path("crowd_data.csv")
HISTORY_COLUMNS = ["timestamp", "people_count", "density", "trend", "estimated_count_5m"]

st.set_page_config(page_title="CrowdSense", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f4f7f5; color: #20332d; }
    [data-testid="stSidebar"] { background: #e8efeb; color: #20332d; }
    [data-testid="stSidebar"] * { color: #20332d; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] { color: #20332d; }
    .hero { padding: 1.1rem 0 0.6rem; }
    .hero h1 { color: #143d35; margin: 0; font-size: 2.25rem; }
    .hero p { color: #536760; margin: 0.25rem 0 0; }
    [data-testid="stMetric"] { background: white; border: 1px solid #dce6e0;
        border-radius: 8px; padding: 1rem; }
    [data-testid="stFileUploader"] section { background: white; border-color: #cbd9d1; }
    [data-testid="stFileUploader"] section * { color: #20332d; }
    .stButton button { color: #143d35; background: white; border-color: #cbd9d1; }
    .stButton button[kind="primary"] { color: white; background: #147d64; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state():
    defaults = {
        "running": False,
        "capture": None,
        "detector": None,
        "history": [],
        "current_count": 0,
        "current_density": "Waiting",
        "trend": "Collecting data",
        "estimate": None,
        "status": "Choose a video source, then press Start.",
        "source_label": "No source selected",
        "temporary_video": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stop_capture():
    capture = st.session_state.capture
    if capture is not None:
        capture.release()
    st.session_state.capture = None
    st.session_state.running = False


def save_count(timestamp: str, count: int, density: str, prediction: dict) -> bool:
    row = {
        "timestamp": timestamp,
        "people_count": count,
        "density": density,
        "trend": prediction["trend"],
        "estimated_count_5m": prediction["estimated_count"],
    }
    st.session_state.history.append(row)
    try:
        pd.DataFrame([row], columns=HISTORY_COLUMNS).to_csv(
            CSV_PATH,
            mode="a",
            header=not CSV_PATH.exists(),
            index=False,
        )
    except OSError as error:
        st.session_state.status = f"Could not save crowd_data.csv: {error}"
        return False
    return True


def start_source(source_mode: str, uploaded_video, confidence: float):
    if source_mode != "Live webcam" and uploaded_video is None:
        st.session_state.status = "Upload a video before pressing Start."
        return

    try:
        with st.spinner("Loading the pretrained YOLO model. The first run may download weights..."):
            detector = CrowdDetector(confidence=confidence)

        if source_mode == "Live webcam":
            capture = open_webcam()
            source_label = "Live webcam"
        else:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            digest = hashlib.sha256(uploaded_video.getvalue()).hexdigest()[:12]
            video_path = Path(tempfile.gettempdir()) / f"crowdsense_{digest}{suffix}"
            video_path.write_bytes(uploaded_video.getvalue())
            capture = open_video(str(video_path))
            st.session_state.temporary_video = str(video_path)
            source_label = uploaded_video.name

        st.session_state.detector = detector
        st.session_state.capture = capture
        st.session_state.running = True
        st.session_state.history = []
        st.session_state.current_count = 0
        st.session_state.current_density = "Waiting"
        st.session_state.trend = "Collecting data"
        st.session_state.estimate = None
        st.session_state.source_label = source_label
        st.session_state.status = "Detection is running. Counts are sampled from real video frames."
    except Exception as error:
        st.session_state.status = f"Could not start detection: {error}"


initialize_state()

with st.sidebar:
    st.subheader("Session settings")
    source_mode = st.radio("Video source", ["Upload video", "Live webcam"])
    uploaded_video = None
    if source_mode == "Upload video":
        uploaded_video = st.file_uploader(
            "Upload Video", type=["mp4", "avi", "mov", "mkv"]
        )
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)

    st.divider()
    st.subheader("Crowd thresholds")
    moderate_max = st.number_input("Low up to", min_value=0, value=10, step=1)
    high_max = st.number_input("Moderate up to", min_value=1, value=25, step=1)
    critical_min = st.number_input("Critical from", min_value=2, value=40, step=1)
    alert_threshold = st.number_input("Alert at people", min_value=1, value=30, step=1)
    if not moderate_max < high_max < critical_min:
        st.warning("Set thresholds in increasing order: Low, Moderate, then Critical.")

    st.divider()
    st.caption(f"CSV log: `{CSV_PATH.resolve()}`")

st.markdown(
    '<div class="hero"><h1>CrowdSense</h1>'
    '<p>Real-Time Crowd Prediction Using AI</p></div>',
    unsafe_allow_html=True,
)

@st.fragment(run_every=0.5)
def live_dashboard():
    start_column, stop_column, reset_column = st.columns([1, 1, 1])
    with start_column:
        if st.button("Start", type="primary", disabled=st.session_state.running):
            if moderate_max < high_max < critical_min:
                start_source(source_mode, uploaded_video, confidence)
            else:
                st.session_state.status = "Fix the crowd thresholds before starting."
    with stop_column:
        if st.button("Stop", disabled=not st.session_state.running):
            stop_capture()
            st.session_state.status = "Detection stopped. The CSV log has been kept."
    with reset_column:
        if st.button("Reset"):
            stop_capture()
            st.session_state.history = []
            st.session_state.current_count = 0
            st.session_state.current_density = "Waiting"
            st.session_state.trend = "Collecting data"
            st.session_state.estimate = None
            st.session_state.source_label = "No source selected"
            st.session_state.status = "Session reset. Existing CSV data was kept."

    if st.session_state.running:
        capture = st.session_state.capture
        success, frame = capture.read() if capture is not None else (False, None)
        if not success or frame is None:
            stop_capture()
            st.session_state.status = "Video ended or a frame could not be read. Press Start to run again."
        else:
            try:
                annotated_frame, count = st.session_state.detector.detect(frame)
                timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                density = classify_density(
                    count,
                    moderate_max=int(moderate_max),
                    high_max=int(high_max),
                    critical_min=int(critical_min),
                )
                prediction = predict_crowd_trend(
                    st.session_state.history + [{"timestamp": timestamp, "people_count": count}]
                )
                saved = save_count(timestamp, count, density, prediction)
                st.session_state.current_count = count
                st.session_state.current_density = density
                st.session_state.trend = prediction["trend"]
                st.session_state.estimate = prediction["estimated_count"]
                if saved:
                    st.session_state.status = "Detection is running. Predictions are estimates from recent counts."
                st.image(annotated_frame, channels="BGR", use_container_width=True)
            except Exception as error:
                stop_capture()
                st.session_state.status = f"Detection stopped because of an error: {error}"

    st.info(st.session_state.status)

    metric_columns = st.columns(4)
    metric_columns[0].metric("People detected", st.session_state.current_count)
    metric_columns[1].metric("Crowd level", st.session_state.current_density)
    metric_columns[2].metric("5-minute trend estimate", st.session_state.trend)
    estimate_label = (
        f"~{st.session_state.estimate} people"
        if st.session_state.estimate is not None
        else "Collecting data"
    )
    metric_columns[3].metric("Estimated count", estimate_label)

    if st.session_state.current_count >= alert_threshold:
        st.warning(
            f"Crowd alert: {st.session_state.current_count} people detected "
            f"(configured alert threshold: {alert_threshold})."
        )

    chart_column, summary_column = st.columns([2, 1])
    with chart_column:
        st.subheader("Crowd count over time")
        history = st.session_state.history
        if history:
            data = pd.DataFrame(history)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            figure, axis = plt.subplots(figsize=(9, 3.5))
            axis.plot(data["timestamp"], data["people_count"], color="#147d64", label="Detected")
            if st.session_state.estimate is not None:
                axis.plot(
                    [data["timestamp"].iloc[-1], data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)],
                    [data["people_count"].iloc[-1], st.session_state.estimate],
                    color="#db7c26",
                    linestyle="--",
                    marker="o",
                    label="5-minute estimate",
                )
            axis.set_ylabel("People")
            axis.set_xlabel("Time")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.autofmt_xdate()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
        else:
            st.caption("Counts and the trend graph appear after detection starts.")

    with summary_column:
        st.subheader("Session")
        st.write(f"**Source:** {st.session_state.source_label}")
        st.write(f"**Samples:** {len(st.session_state.history)}")
        st.write("**Prediction:** Estimate based on recent detected counts; not a guarantee.")
        if CSV_PATH.exists():
            try:
                csv_data = CSV_PATH.read_bytes()
                st.download_button(
                    "Download crowd_data.csv",
                    data=csv_data,
                    file_name="crowd_data.csv",
                    mime="text/csv",
                )
            except OSError as error:
                st.caption(f"CSV download unavailable: {error}")


live_dashboard()