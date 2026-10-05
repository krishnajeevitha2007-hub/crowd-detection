"""YOLO-based people detection for CrowdSense."""

import cv2


class CrowdDetector:
    """Load a pretrained YOLO model and count people in video frames."""

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Run: py -m pip install -r requirements.txt"
            ) from error

        # Ultralytics downloads this official pretrained weight on first use.
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """Return an annotated frame and the number of detected people."""
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )
        result = results[0]
        annotated_frame = result.plot()
        people_count = 0 if result.boxes is None else len(result.boxes)
        return annotated_frame, people_count


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and report a useful error when it cannot be read."""
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        video.release()
        raise RuntimeError(
            f"Could not open the video: {path}. Try a different MP4, AVI, or MOV file."
        )
    return video


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Open the default local webcam."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "The webcam could not be opened. Check camera permissions, close other "
            "apps using it, or choose Upload video instead."
        )
    return camera