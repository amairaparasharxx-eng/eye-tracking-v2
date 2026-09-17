"""Turn raw landmarks into physically meaningful numbers.

Everything is scaled by the iris, so results are in millimetres and are
insensitive to how far the person sits from the camera. Head rotation is
estimated separately so that eye-in-head position can be reported apart
from gaze direction in space.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from .landmarks import (
    IRIS_DIAMETER_MM,
    LEFT_EYE,
    POSE_LANDMARKS,
    POSE_MODEL_3D,
    RIGHT_EYE,
    assign_iris,
)

# A blink or near-blink invalidates lid-height measurements, so those frames
# are dropped from lid analyses rather than averaged in.
BLINK_EAR_THRESHOLD = 0.15
CLOSED_EAR_THRESHOLD = 0.10


@dataclass
class EyeMeasure:
    """Measurements for one eye in one frame."""

    mrd1_mm: float = float("nan")        # upper lid margin -> pupil centre
    mrd2_mm: float = float("nan")        # pupil centre -> lower lid margin
    fissure_mm: float = float("nan")     # total vertical opening
    ear: float = float("nan")            # eye aspect ratio, unitless
    iris_visible_frac: float = float("nan")  # how much iris the lids leave
    gaze_h: float = float("nan")         # eye-in-head, -1 = fully in, +1 = out
    gaze_v: float = float("nan")         # eye-in-head vertical, + = up
    iris_x_mm: float = float("nan")      # iris centre in orbit, mm, + = out
    iris_y_mm: float = float("nan")      # iris centre in orbit, mm, + = up
    brow_to_lid_mm: float = float("nan")  # frontalis recruitment proxy
    closed: bool = False
    blinking: bool = False


@dataclass
class FrameMeasure:
    """All measurements for one frame."""

    t: float = 0.0
    valid: bool = False
    right: EyeMeasure = field(default_factory=EyeMeasure)
    left: EyeMeasure = field(default_factory=EyeMeasure)
    head_roll_deg: float = float("nan")   # + = head tilted to subject's left
    head_yaw_deg: float = float("nan")    # + = turned to subject's left
    head_pitch_deg: float = float("nan")  # + = chin up
    face_width_px: float = float("nan")
    mm_per_px: float = float("nan")
    quality: float = 0.0                  # 0..1 crude acquisition quality
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["right"] = asdict(self.right)
        d["left"] = asdict(self.left)
        return d


def _fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic circle fit through the iris ring points."""
    x, y = pts[:, 0], pts[:, 1]
    a_mat = np.column_stack([x, y, np.ones(len(x))])
    b_vec = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        centre = pts.mean(axis=0)
        return centre, float(np.linalg.norm(pts - centre, axis=1).mean())
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r_sq = sol[2] + cx ** 2 + cy ** 2
    r = float(np.sqrt(max(r_sq, 1e-9)))
    return np.array([cx, cy]), r


def _ear(points: np.ndarray, spec: dict) -> float:
    """Eye aspect ratio: vertical opening divided by horizontal width."""
    p_out = points[spec["outer_corner"]]
    p_in = points[spec["inner_corner"]]
    width = float(np.linalg.norm(p_out - p_in))
    if width < 1e-6:
        return float("nan")
    upper = points[spec["upper_lid"]]
    lower = points[spec["lower_lid"]]
    n = min(len(upper), len(lower))
    gaps = [float(np.linalg.norm(upper[i] - lower[i])) for i in range(n)]
    return float(np.mean(gaps) / width)


def _lid_y_at_x(points: np.ndarray, indices: list[int], x: float) -> float:
    """Vertical position of a lid margin directly above/below a given x.

    Interpolated along the lid contour rather than taking the single mid
    point, so the measurement stays correct when the eye looks sideways
    and the pupil is no longer under the middle of the lid.
    """
    contour = points[indices]
    order = np.argsort(contour[:, 0])
    xs = contour[order, 0]
    ys = contour[order, 1]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def _wrap_small_angle(deg: float) -> float:
    """Map an angle into (-90, 90], for quantities known to be physically small."""
    if not np.isfinite(deg):
        return float(deg)
    deg = float((deg + 180.0) % 360.0 - 180.0)
    if deg > 90.0:
        deg -= 180.0
    elif deg <= -90.0:
        deg += 180.0
    return deg


def _euler_from_pose(points: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Head rotation from a generic 3D head model via solvePnP."""
    image_pts = points[POSE_LANDMARKS].astype(np.float64)
    focal = float(w)
    cam = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros((4, 1))
    ok, rvec, _tvec = cv2.solvePnP(
        POSE_MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return float("nan"), float("nan"), float("nan")

    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2, 1], rmat[2, 2]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = np.degrees(np.arctan2(-rmat[1, 0], rmat[0, 0]))
    else:  # gimbal-locked, rare
        pitch = np.degrees(np.arctan2(rmat[1, 2], rmat[1, 1]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = 0.0

    # solvePnP returns roll and pitch near +-180 for an upright face, because
    # the model's y axis points up while image y points down. Physical head
    # roll and pitch are small, so wrap both into a sane range - otherwise a
    # 10 degree tilt is reported as a 350 degree one.
    pitch = _wrap_small_angle(pitch)
    roll = _wrap_small_angle(roll)
    return float(roll), float(yaw), float(pitch)


def _measure_eye(
    points: np.ndarray,
    spec: dict,
    iris_idx: list[int],
    mm_per_px: float,
    outward_sign: float,
) -> EyeMeasure:
    m = EyeMeasure()

    iris_pts = points[iris_idx]
    centre, radius_px = _fit_circle(iris_pts)
    if radius_px < 1.0:
        return m

    upper_y = _lid_y_at_x(points, spec["upper_lid"], centre[0])
    lower_y = _lid_y_at_x(points, spec["lower_lid"], centre[0])

    # Image y grows downward, so the upper lid has the smaller y.
    m.mrd1_mm = float((centre[1] - upper_y) * mm_per_px)
    m.mrd2_mm = float((lower_y - centre[1]) * mm_per_px)
    m.fissure_mm = m.mrd1_mm + m.mrd2_mm
    m.ear = _ear(points, spec)

    iris_r_mm = radius_px * mm_per_px
    if iris_r_mm > 0:
        exposed = np.clip(m.mrd1_mm, 0, iris_r_mm) + np.clip(m.mrd2_mm, 0, iris_r_mm)
        m.iris_visible_frac = float(np.clip(exposed / (2 * iris_r_mm), 0.0, 1.0))

    corner_out = points[spec["outer_corner"]]
    corner_in = points[spec["inner_corner"]]
    eye_width = float(np.linalg.norm(corner_out - corner_in))
    corner_mid = 0.5 * (corner_out + corner_in)
    if eye_width > 1e-6:
        # Horizontal position of the iris within the palpebral aperture,
        # signed so that + always means "towards the temple" for both eyes.
        m.gaze_h = float(outward_sign * (centre[0] - corner_mid[0]) / (eye_width / 2))
        # Vertical reference is the intercanthal line, NOT the lid midpoint.
        # The upper lid follows the globe on vertical gaze, so measuring against
        # the lids would cancel out most of the movement being measured.
        m.gaze_v = float((corner_mid[1] - centre[1]) / (eye_width / 2))
    m.iris_x_mm = float(outward_sign * (centre[0] - corner_mid[0]) * mm_per_px)
    m.iris_y_mm = float((corner_mid[1] - centre[1]) * mm_per_px)

    brow_y = _lid_y_at_x(points, spec["brow"], centre[0])
    m.brow_to_lid_mm = float((upper_y - brow_y) * mm_per_px)

    if not np.isnan(m.ear):
        m.closed = m.ear < CLOSED_EAR_THRESHOLD
        m.blinking = m.ear < BLINK_EAR_THRESHOLD
    return m


def measure_frame(face_frame, prev: FrameMeasure | None = None) -> FrameMeasure:
    """Compute all measurements for one tracked frame."""
    fm = FrameMeasure(t=face_frame.timestamp)
    if not face_frame.ok or face_frame.points is None:
        fm.note = face_frame.reason or "no face"
        return fm

    pts = face_frame.points
    right_iris, left_iris = assign_iris(pts)

    _, r_right = _fit_circle(pts[right_iris])
    _, r_left = _fit_circle(pts[left_iris])
    radii = [r for r in (r_right, r_left) if r > 1.0]
    if not radii:
        fm.note = "iris not resolved"
        return fm
    mm_per_px = IRIS_DIAMETER_MM / (2.0 * float(np.mean(radii)))
    fm.mm_per_px = mm_per_px

    # outward_sign: +1 means increasing image-x is towards that eye's temple.
    # Subject's right eye sits on the image left, so its temple is at low x.
    fm.right = _measure_eye(pts, RIGHT_EYE, right_iris, mm_per_px, -1.0)
    fm.left = _measure_eye(pts, LEFT_EYE, left_iris, mm_per_px, +1.0)

    roll, yaw, pitch = _euler_from_pose(pts, face_frame.width, face_frame.height)
    fm.head_roll_deg, fm.head_yaw_deg, fm.head_pitch_deg = roll, yaw, pitch
    fm.face_width_px = float(
        np.linalg.norm(pts[RIGHT_EYE["outer_corner"]] - pts[LEFT_EYE["outer_corner"]])
    )

    fm.quality = _quality(fm, face_frame)
    fm.valid = True
    return fm


def _quality(fm: FrameMeasure, face_frame) -> float:
    """Crude 0..1 acquisition quality for this frame.

    Penalises a face that is too small in frame, strongly turned away, or
    producing implausible iris scaling.
    """
    score = 1.0
    if not np.isnan(fm.face_width_px) and face_frame.width:
        frac = fm.face_width_px / face_frame.width
        if frac < 0.15:
            score *= 0.4
        elif frac < 0.22:
            score *= 0.75
    if not np.isnan(fm.head_yaw_deg):
        score *= float(np.clip(1.0 - abs(fm.head_yaw_deg) / 45.0, 0.2, 1.0))
    if not np.isnan(fm.mm_per_px):
        # An iris should be roughly 30-160 px across at usable resolutions.
        if not (0.03 < fm.mm_per_px < 0.45):
            score *= 0.3
    return float(np.clip(score, 0.0, 1.0))
