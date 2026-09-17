"""Face landmark extraction.

Supports both MediaPipe front-ends:
  * the current Tasks API (mediapipe.tasks.python.vision.FaceLandmarker),
    which needs a downloaded .task model file
  * the older solutions API (mediapipe.solutions.face_mesh), which bundles
    its own model

Both give 478 landmarks when iris refinement is on: 468 face points plus
two 5-point iris rings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------
# Landmark indices (MediaPipe canonical face mesh).
# MediaPipe labels these anatomically: "right" = the subject's right eye,
# which appears on the LEFT of an unmirrored image.
# ----------------------------------------------------------------------------

RIGHT_EYE = {
    "outer_corner": 33,
    "inner_corner": 133,
    "upper_lid": [246, 161, 160, 159, 158, 157, 173],
    "upper_lid_mid": 159,
    "lower_lid": [7, 163, 144, 145, 153, 154, 155],
    "lower_lid_mid": 145,
    "brow": [70, 63, 105, 66, 107],
    "brow_mid": 105,
}

LEFT_EYE = {
    "outer_corner": 263,
    "inner_corner": 362,
    "upper_lid": [466, 388, 387, 386, 385, 384, 398],
    "upper_lid_mid": 386,
    "lower_lid": [249, 390, 373, 374, 380, 381, 382],
    "lower_lid_mid": 374,
    "brow": [300, 293, 334, 296, 336],
    "brow_mid": 334,
}

# The two 5-point iris rings added by refinement. Which ring belongs to which
# eye is NOT assumed here - assign_iris() works it out geometrically, because
# the convention has differed between MediaPipe releases.
IRIS_RING_A = [468, 469, 470, 471, 472]
IRIS_RING_B = [473, 474, 475, 476, 477]

# Points used for head pose, with a generic 3D head model (millimetres).
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
POSE_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # 1   nose tip
        [0.0, -63.6, -12.5],      # 152 chin
        [-43.3, 32.7, -26.0],     # 33  subject right eye, outer corner
        [43.3, 32.7, -26.0],      # 263 subject left eye, outer corner
        [-28.9, -28.9, -24.1],    # 61  subject right mouth corner
        [28.9, -28.9, -24.1],     # 291 subject left mouth corner
    ],
    dtype=np.float64,
)

# Human iris horizontal diameter is remarkably constant across adults.
# Used as the only scale reference, so all outputs are in millimetres and
# do not change when the person moves nearer to or further from the camera.
IRIS_DIAMETER_MM = 11.7

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceFrame:
    """One frame of tracking output."""

    ok: bool
    points: np.ndarray | None = None      # (N, 2) pixel coordinates
    points3d: np.ndarray | None = None    # (N, 3) normalised, z is relative
    width: int = 0
    height: int = 0
    timestamp: float = 0.0
    reason: str = ""                      # why ok is False


class FaceTracker:
    """Thin wrapper that hides which MediaPipe API is installed."""

    def __init__(self, model_path: str | None = None):
        self.backend = None
        self._impl = None
        self._frame_index = 0
        self._init_backend(model_path)

    # -- setup ---------------------------------------------------------------

    def _init_backend(self, model_path: str | None) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            path = model_path or self._find_model()
            if path is None:
                raise FileNotFoundError(
                    "Face landmark model not found. Run 'python download_model.py'."
                )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._impl = mp_vision.FaceLandmarker.create_from_options(options)
            self.backend = "tasks"
            return
        except FileNotFoundError:
            raise
        except Exception:
            pass

        try:
            import mediapipe as mp

            self._impl = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.backend = "solutions"
            return
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Could not start MediaPipe face tracking. Install it with:\n"
                "    pip install mediapipe\n"
                f"Underlying error: {exc}"
            ) from exc

    @staticmethod
    def _find_model() -> str | None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for candidate in (
            os.path.join(here, MODEL_FILENAME),
            os.path.join(here, "models", MODEL_FILENAME),
            os.path.join(os.getcwd(), MODEL_FILENAME),
        ):
            if os.path.isfile(candidate):
                return candidate
        return None

    # -- per frame -----------------------------------------------------------

    def process(self, bgr_frame: np.ndarray, timestamp: float) -> FaceFrame:
        h, w = bgr_frame.shape[:2]
        rgb = bgr_frame[:, :, ::-1].copy()

        if self.backend == "tasks":
            import mediapipe as mp

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_index += 1
            result = self._impl.detect_for_video(image, int(timestamp * 1000))
            faces = result.face_landmarks
        else:
            result = self._impl.process(rgb)
            faces = result.multi_face_landmarks

        if not faces:
            return FaceFrame(
                ok=False, width=w, height=h, timestamp=timestamp,
                reason="no face found",
            )

        lms = faces[0]
        if self.backend == "tasks":
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
        else:
            arr = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float64)

        if arr.shape[0] < 478:
            return FaceFrame(
                ok=False, width=w, height=h, timestamp=timestamp,
                reason="iris landmarks missing (refine_landmarks off?)",
            )

        px = np.column_stack([arr[:, 0] * w, arr[:, 1] * h])
        return FaceFrame(
            ok=True, points=px, points3d=arr, width=w, height=h,
            timestamp=timestamp,
        )

    def close(self) -> None:
        try:
            self._impl.close()
        except Exception:
            pass


def assign_iris(points: np.ndarray) -> tuple[list[int], list[int]]:
    """Return (right_eye_iris_indices, left_eye_iris_indices).

    Decided by distance to each eye's corner midpoint rather than by a
    hard-coded convention, because the ordering of the two iris rings has
    not been stable across MediaPipe versions and getting it backwards
    would silently swap every left/right result in the whole program.
    """
    right_centre = 0.5 * (
        points[RIGHT_EYE["outer_corner"]] + points[RIGHT_EYE["inner_corner"]]
    )
    ring_a_centre = points[IRIS_RING_A].mean(axis=0)
    ring_b_centre = points[IRIS_RING_B].mean(axis=0)

    d_a = float(np.linalg.norm(ring_a_centre - right_centre))
    d_b = float(np.linalg.norm(ring_b_centre - right_centre))
    if d_a <= d_b:
        return IRIS_RING_A, IRIS_RING_B
    return IRIS_RING_B, IRIS_RING_A
