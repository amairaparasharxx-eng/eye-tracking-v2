"""Saving sessions and comparing them across visits.

Only numbers are written. Raw frames never leave memory unless the user
explicitly turns on clip saving in the UI.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np

from . import signals

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "EyeSignRecorder")


def _sanitise(obj):
    """Make results JSON-safe: NaN and numpy types are not."""
    if isinstance(obj, dict):
        return {str(k): _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitise(obj.tolist())
    return obj


def ensure_dir(path: str | None = None) -> str:
    path = path or DEFAULT_DIR
    os.makedirs(path, exist_ok=True)
    return path


def save_session(session: dict, directory: str | None = None) -> str:
    directory = ensure_dir(directory)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    person = session.get("person_label", "session").strip() or "session"
    safe = "".join(c for c in person if c.isalnum() or c in "-_") or "session"
    path = os.path.join(directory, f"{safe}_{stamp}.json")
    # Two sessions saved in the same second must not overwrite each other.
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{safe}_{stamp}_{suffix}.json")
        suffix += 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitise(session), fh, indent=2)
    return path


def load_sessions(directory: str | None = None, person_label: str | None = None) -> list[dict]:
    directory = directory or DEFAULT_DIR
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if person_label and data.get("person_label") != person_label:
            continue
        data["_file"] = name
        out.append(data)
    out.sort(key=lambda d: d.get("started_at", ""))
    return out


# Which single number best represents each test, for cross-visit tracking.
TRACKED_METRICS = {
    "fatigable_ptosis": [
        ("results.per_eye.right.change_mm", "Right lid change on upgaze (mm)"),
        ("results.per_eye.left.change_mm", "Left lid change on upgaze (mm)"),
        ("results.per_eye.right.baseline_mrd1_mm", "Right resting lid height (mm)"),
        ("results.per_eye.left.baseline_mrd1_mm", "Left resting lid height (mm)"),
    ],
    "cogan_lid_twitch": [
        ("results.per_eye.right.overshoot_mm", "Right lid overshoot (mm)"),
        ("results.per_eye.left.overshoot_mm", "Left lid overshoot (mm)"),
    ],
    "curtain_sign": [
        ("results.contralateral_change_mm", "Other lid change while one held up (mm)"),
    ],
    "peek_sign": [
        ("results.per_eye.right.reopening_mm", "Right eye reopening (mm)"),
        ("results.per_eye.left.reopening_mm", "Left eye reopening (mm)"),
    ],
    "ophthalmoparesis": [
        ("results.max_interocular_difference", "Largest left/right excursion difference"),
    ],
    "fatigable_saccades": [
        ("results.per_eye.right.amplitude_decrement_pct", "Right saccade size decrement (%)"),
        ("results.per_eye.left.amplitude_decrement_pct", "Left saccade size decrement (%)"),
    ],
    "gaze_holding": [
        ("results.holds.hold_right.right.drift_rate_per_s", "Right eye drift, gaze right"),
        ("results.holds.hold_left.left.drift_rate_per_s", "Left eye drift, gaze left"),
    ],
    "head_posture": [
        ("results.roll.offset_deg", "Head tilt offset (deg)"),
        ("results.yaw.offset_deg", "Head turn offset (deg)"),
    ],
    "variability": [
        ("results.per_eye.right.range", "Right lid range within exam (mm)"),
        ("results.per_eye.left.range", "Left lid range within exam (mm)"),
    ],
}


def _dig(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def compare_sessions(sessions: list[dict]) -> dict:
    """Assemble each tracked metric across visits, with variability stats."""
    series: dict[str, dict] = {}
    dates = [s.get("started_at", "") for s in sessions]

    for test_key, metrics in TRACKED_METRICS.items():
        for dotted, label in metrics:
            values = []
            for s in sessions:
                test = (s.get("tests") or {}).get(test_key)
                values.append(_dig(test, dotted) if test else None)
            if not any(v is not None for v in values):
                continue
            numeric = [v for v in values if isinstance(v, (int, float))]
            series[f"{test_key}::{dotted}"] = {
                "test": test_key,
                "label": label,
                "dates": dates,
                "values": values,
                "summary": signals.summarise(np.asarray(numeric, float)),
                "cov": signals.coefficient_of_variation(numeric),
            }
    return {
        "n_sessions": len(sessions),
        "dates": dates,
        "series": series,
        "note": (
            "Differences between visits reflect the camera, the lighting, the "
            "time of day and how the person sat, as much as anything else. "
            "Do not read a trend here as a change in health."
        ),
    }
