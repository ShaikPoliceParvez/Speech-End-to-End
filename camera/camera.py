import cv2
from datetime import datetime
from pathlib import Path

from config import CAMERA_INDEX, CAPTURE_SAVE_IMAGES, CAPTURE_MAX_FILES


class Camera:

    def _prune_old_captures(self, capture_dir):
        if CAPTURE_MAX_FILES <= 0:
            return

        files = sorted(
            capture_dir.glob("capture_*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for stale in files[CAPTURE_MAX_FILES:]:
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass

    def capture(self):

        cap = cv2.VideoCapture(CAMERA_INDEX)

        ok, frame = cap.read()

        cap.release()

        if not ok:
            raise RuntimeError("Camera Capture Failed")

        if not CAPTURE_SAVE_IMAGES:
            ok_encode, encoded = cv2.imencode(".jpg", frame)
            if not ok_encode:
                raise RuntimeError("Failed to encode captured image")
            return encoded.tobytes(), {"width": int(frame.shape[1]), "height": int(frame.shape[0])}

        capture_dir = Path(__file__).resolve().parent.parent / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)

        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = capture_dir / filename

        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError("Failed to save captured image")

        self._prune_old_captures(capture_dir)

        return str(image_path), {"width": int(frame.shape[1]), "height": int(frame.shape[0])}