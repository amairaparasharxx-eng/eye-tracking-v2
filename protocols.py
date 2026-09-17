"""The nine observation protocols.

Each protocol is a sequence of phases. A phase has a plain-language
instruction, a duration, and optionally an on-screen fixation target. After
recording, the protocol's analyser turns the collected frames into numbers,
a quality judgement, and a plain-language description of what was recorded.

The description NEVER says whether a sign is present or absent in a clinical
sense. It describes the recording. That distinction is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import signals
from .disclaimers import TEST_CAVEATS
from .metrics import FrameMeasure

# Fixation target positions, as fractions of the preview area.
CENTRE = (0.5, 0.5)
UP = (0.5, 0.06)
DOWN = (0.5, 0.94)
LEFT = (0.06, 0.5)
RIGHT = (0.94, 0.5)
UP_LEFT = (0.08, 0.1)
UP_RIGHT = (0.92, 0.1)
DOWN_LEFT = (0.08, 0.9)
DOWN_RIGHT = (0.92, 0.9)


@dataclass
class Phase:
    key: str
    instruction: str
    duration_s: float
    target: tuple[float, float] | None = None
    record: bool = True
    countdown: bool = True
    # Set when the phase needs a second person in the room.
    helper_action: str = ""


@dataclass
class Protocol:
    key: str
    title: str
    plain_title: str
    why: str
    needs_helper: bool
    phases: list[Phase]
    caveat_key: str
    # Attached after the analyser functions are defined, at the bottom of this module.
    analyse: Callable[[dict[str, list[FrameMeasure]]], dict] | None = None
    min_fps: float = 15.0
    repeats: int = 1


# ---------------------------------------------------------------------------
# helpers shared by the analysers
# ---------------------------------------------------------------------------


def _series(frames: list[FrameMeasure], eye: str, attr: str,
            drop_blinks: bool = True) -> tuple[np.ndarray, np.ndarray]:
    ts, ys = [], []
    for f in frames:
        if not f.valid:
            continue
        e = getattr(f, eye)
        if drop_blinks and e.blinking:
            continue
        ts.append(f.t)
        ys.append(getattr(e, attr))
    return np.asarray(ts, float), np.asarray(ys, float)


def _mean_quality(frames: list[FrameMeasure]) -> float:
    q = [f.quality for f in frames if f.valid]
    return float(np.mean(q)) if q else 0.0


def _tracking_rate(frames: list[FrameMeasure]) -> float:
    if not frames:
        return 0.0
    return float(sum(1 for f in frames if f.valid) / len(frames))


def _quality_block(frames: list[FrameMeasure], min_fps: float) -> dict:
    ts = np.asarray([f.t for f in frames if f.valid], float)
    fps = signals.estimate_fps(ts)
    rate = _tracking_rate(frames)
    mean_q = _mean_quality(frames)
    problems = []
    if rate < 0.8:
        problems.append(f"face lost in {(1 - rate) * 100:.0f}% of frames")
    if np.isfinite(fps) and fps < min_fps:
        problems.append(f"camera only managed {fps:.0f} frames per second")
    if mean_q < 0.6:
        problems.append("face too small, too dark, or turned away")
    usable = rate >= 0.7 and mean_q >= 0.45 and (not np.isfinite(fps) or fps >= 8)
    return {
        "fps": fps,
        "tracking_rate": rate,
        "mean_quality": mean_q,
        "problems": problems,
        "usable": bool(usable),
    }


def _head_moved(frames: list[FrameMeasure], limit_deg: float = 8.0) -> dict:
    yaw = np.asarray([f.head_yaw_deg for f in frames if f.valid], float)
    pitch = np.asarray([f.head_pitch_deg for f in frames if f.valid], float)
    roll = np.asarray([f.head_roll_deg for f in frames if f.valid], float)
    out = {}
    for name, arr in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
        arr = arr[np.isfinite(arr)]
        out[f"{name}_range_deg"] = float(np.ptp(arr)) if arr.size else float("nan")
    ranges = [v for v in out.values() if np.isfinite(v)]
    out["head_moved"] = bool(ranges and max(ranges) > limit_deg)
    return out


# ---------------------------------------------------------------------------
# 1. Fatigable ptosis
# ---------------------------------------------------------------------------

def analyse_fatigable_ptosis(phases: dict[str, list[FrameMeasure]]) -> dict:
    base = phases.get("baseline", [])
    hold = phases.get("upgaze", [])
    rest = phases.get("recovery", [])
    res = {"per_eye": {}}

    for eye in ("right", "left"):
        tb, yb = _series(base, eye, "mrd1_mm")
        th, yh = _series(hold, eye, "mrd1_mm")
        tr, yr = _series(rest, eye, "mrd1_mm")
        tr_res = signals.trend(th, signals.median_filter(yh, 5),
                               window_s=5.0, drop_threshold=1.0)
        baseline_mm = float(np.median(yb)) if yb.size else float("nan")
        recovery_mm = float(np.median(yr)) if yr.size else float("nan")
        _, brow = _series(hold, eye, "brow_to_lid_mm")
        brow_trend = signals.trend(th, signals.median_filter(brow, 5)) if brow.size else None

        res["per_eye"][eye] = {
            "baseline_mrd1_mm": baseline_mm,
            "start_mm": tr_res.start_mean,
            "end_mm": tr_res.end_mean,
            "change_mm": tr_res.change,
            "slope_mm_per_min": tr_res.slope_per_min,
            "r2": tr_res.r2,
            "time_to_1mm_drop_s": tr_res.time_to_drop,
            "recovery_after_rest_mm": recovery_mm,
            "recovered_fraction": (
                float((recovery_mm - tr_res.end_mean) /
                      (baseline_mm - tr_res.end_mean))
                if all(np.isfinite(v) for v in (recovery_mm, baseline_mm, tr_res.end_mean))
                and abs(baseline_mm - tr_res.end_mean) > 0.2 else float("nan")
            ),
            "frontalis_change_mm": brow_trend.change if brow_trend else float("nan"),
        }

    r = res["per_eye"]["right"]["change_mm"]
    l = res["per_eye"]["left"]["change_mm"]
    res["asymmetry_mm"] = float(abs(r - l)) if np.isfinite(r) and np.isfinite(l) else float("nan")
    res["quality"] = _quality_block(hold, 15.0)
    res["head"] = _head_moved(hold)

    parts = []
    for eye, label in (("right", "right eye"), ("left", "left eye")):
        d = res["per_eye"][eye]
        if np.isfinite(d["change_mm"]):
            direction = "fell" if d["change_mm"] < 0 else "rose"
            parts.append(
                f"During one minute of looking up, the upper eyelid margin of the "
                f"{label} {direction} by {abs(d['change_mm']):.1f} mm "
                f"({d['start_mm']:.1f} mm at the start, {d['end_mm']:.1f} mm at the end)."
            )
    if np.isfinite(res["asymmetry_mm"]):
        parts.append(f"The two sides differed by {res['asymmetry_mm']:.1f} mm in how much they changed.")
    res["description"] = " ".join(parts) or "Not enough usable frames to describe."
    return res


# ---------------------------------------------------------------------------
# 2. Cogan's lid twitch
# ---------------------------------------------------------------------------

def analyse_cogan(phases: dict[str, list[FrameMeasure]]) -> dict:
    down = phases.get("downgaze", [])
    back = phases.get("refixate", [])
    res = {"per_eye": {}}

    for eye in ("right", "left"):
        t, y = _series(back, eye, "mrd1_mm")
        _, blink_y = _series(back, eye, "mrd1_mm", drop_blinks=False)
        detail = {
            "overshoot_mm": float("nan"),
            "overshoot_latency_s": float("nan"),
            "settled_mm": float("nan"),
            "peak_mm": float("nan"),
            "n_samples": int(t.size),
        }
        if t.size >= 8:
            t0 = t - t[0]
            ys = signals.median_filter(y, 3)
            window = (t0 >= 0.05) & (t0 <= 1.5)
            settle = ys[t0 >= 1.2]
            if window.any() and settle.size >= 3:
                peak_i = int(np.argmax(ys[window]))
                peak = float(ys[window][peak_i])
                settled = float(np.median(settle))
                detail["peak_mm"] = peak
                detail["settled_mm"] = settled
                detail["overshoot_mm"] = peak - settled
                detail["overshoot_latency_s"] = float(t0[window][peak_i])
        res["per_eye"][eye] = detail

    res["quality"] = _quality_block(back, 25.0)
    res["downgaze_seconds"] = float(down[-1].t - down[0].t) if len(down) > 2 else float("nan")
    fps = res["quality"]["fps"]
    res["timing_reliable"] = bool(np.isfinite(fps) and fps >= 45)

    parts = []
    for eye, label in (("right", "right"), ("left", "left")):
        d = res["per_eye"][eye]
        if np.isfinite(d["overshoot_mm"]):
            parts.append(
                f"After looking back up, the {label} upper eyelid reached a position "
                f"{d['overshoot_mm']:.1f} mm above where it settled, "
                f"{d['overshoot_latency_s'] * 1000:.0f} ms after refixation."
            )
    if not res["timing_reliable"]:
        parts.append(
            "The camera was too slow to time a lid twitch properly, so these "
            "numbers cannot be trusted either way."
        )
    res["description"] = " ".join(parts) or "Not enough usable frames to describe."
    return res


# ---------------------------------------------------------------------------
# 3. Curtain sign / enhanced ptosis
# ---------------------------------------------------------------------------

def analyse_curtain(phases: dict[str, list[FrameMeasure]]) -> dict:
    base = phases.get("baseline", [])
    lifted = phases.get("lifted", [])
    res = {"per_eye": {}, "manipulated_eye": None}

    changes = {}
    for eye in ("right", "left"):
        _, yb = _series(base, eye, "mrd1_mm")
        _, yl = _series(lifted, eye, "mrd1_mm")
        b = float(np.median(yb)) if yb.size else float("nan")
        a = float(np.median(yl)) if yl.size else float("nan")
        changes[eye] = a - b
        res["per_eye"][eye] = {
            "baseline_mrd1_mm": b,
            "during_lift_mrd1_mm": a,
            "change_mm": a - b,
        }

    # The eye whose lid was physically raised shows a large upward jump.
    finite = {k: v for k, v in changes.items() if np.isfinite(v)}
    if finite:
        lifted_eye = max(finite, key=lambda k: finite[k])
        if finite[lifted_eye] > 0.8:
            res["manipulated_eye"] = lifted_eye
            other = "left" if lifted_eye == "right" else "right"
            res["contralateral_change_mm"] = changes[other]
            res["contralateral_eye"] = other

    res["quality"] = _quality_block(lifted, 15.0)
    if res["manipulated_eye"] is None:
        res["description"] = (
            "No eyelid was detected as having been lifted, so nothing can be "
            "compared. Repeat with a helper holding one upper lid raised."
        )
    else:
        other = res["contralateral_eye"]
        change = res["contralateral_change_mm"]
        direction = "fell" if change < 0 else "rose"
        res["description"] = (
            f"While the {res['manipulated_eye']} upper lid was held up, the "
            f"{other} upper lid {direction} by {abs(change):.1f} mm."
        )
    return res


# ---------------------------------------------------------------------------
# 4. Peek sign
# ---------------------------------------------------------------------------

def analyse_peek(phases: dict[str, list[FrameMeasure]]) -> dict:
    close = phases.get("gentle_closure", [])
    res = {"per_eye": {}}

    for eye in ("right", "left"):
        ts, ys = [], []
        for f in close:
            if not f.valid:
                continue
            ts.append(f.t)
            ys.append(getattr(f, eye).fissure_mm)
        t = np.asarray(ts, float)
        y = signals.median_filter(np.asarray(ys, float), 5)
        detail = {
            "closure_achieved": False,
            "time_to_closure_s": float("nan"),
            "reopening_mm": float("nan"),
            "time_to_reopening_s": float("nan"),
            "final_fissure_mm": float("nan"),
            "min_fissure_mm": float("nan"),
        }
        good = np.isfinite(t) & np.isfinite(y)
        t, y = t[good], y[good]
        if t.size >= 20:
            t0 = t - t[0]
            closed_level = float(np.min(y))
            detail["min_fissure_mm"] = closed_level
            closed_idx = np.where(y <= closed_level + 0.4)[0]
            if closed_idx.size and closed_level < 2.0:
                detail["closure_achieved"] = True
                i0 = int(closed_idx[0])
                detail["time_to_closure_s"] = float(t0[i0])
                after = y[i0:]
                t_after = t0[i0:]
                if after.size > 5:
                    reopening = after - closed_level
                    j = int(np.argmax(reopening))
                    detail["reopening_mm"] = float(reopening[j])
                    detail["time_to_reopening_s"] = float(t_after[j] - t0[i0])
                    detail["final_fissure_mm"] = float(np.median(after[-10:]))
        res["per_eye"][eye] = detail

    res["quality"] = _quality_block(close, 12.0)
    parts = []
    for eye, label in (("right", "right"), ("left", "left")):
        d = res["per_eye"][eye]
        if d["closure_achieved"] and np.isfinite(d["reopening_mm"]):
            parts.append(
                f"The {label} eye closed, then opened again by "
                f"{d['reopening_mm']:.1f} mm over the following "
                f"{d['time_to_reopening_s']:.0f} seconds."
            )
        elif not d["closure_achieved"]:
            parts.append(f"The {label} eye never fully closed, so this item is void.")
    res["description"] = " ".join(parts) or "Not enough usable frames to describe."
    return res


# ---------------------------------------------------------------------------
# 5. Variable / asymmetric ophthalmoparesis
# ---------------------------------------------------------------------------

_GAZE_PHASES = [
    ("right_gaze", "gaze_h", +1),
    ("left_gaze", "gaze_h", +1),
    ("up_gaze", "gaze_v", +1),
    ("down_gaze", "gaze_v", -1),
]


def analyse_ophthalmoparesis(phases: dict[str, list[FrameMeasure]]) -> dict:
    res = {"positions": {}, "per_eye": {"right": {}, "left": {}}}
    centre = phases.get("centre", [])

    ref = {}
    for eye in ("right", "left"):
        _, h = _series(centre, eye, "gaze_h")
        _, v = _series(centre, eye, "gaze_v")
        ref[eye] = (
            float(np.median(h)) if h.size else 0.0,
            float(np.median(v)) if v.size else 0.0,
        )

    for phase_key, attr, _sign in _GAZE_PHASES:
        frames = phases.get(phase_key, [])
        if not frames:
            continue
        entry = {}
        for eye in ("right", "left"):
            _, y = _series(frames, eye, attr)
            base = ref[eye][0] if attr == "gaze_h" else ref[eye][1]
            excursion = float(np.median(y) - base) if y.size else float("nan")
            entry[eye] = {
                "excursion": excursion,
                "steadiness_sd": float(np.std(y)) if y.size > 2 else float("nan"),
            }
        r, l = entry["right"]["excursion"], entry["left"]["excursion"]
        entry["interocular_difference"] = (
            float(abs(abs(r) - abs(l))) if np.isfinite(r) and np.isfinite(l) else float("nan")
        )
        entry["head"] = _head_moved(frames, limit_deg=6.0)
        entry["quality"] = _quality_block(frames, 12.0)
        res["positions"][phase_key] = entry

    diffs = [e["interocular_difference"] for e in res["positions"].values()]
    res["max_interocular_difference"] = (
        float(np.nanmax(diffs)) if any(np.isfinite(d) for d in diffs) else float("nan")
    )
    res["head_contaminated"] = any(
        e.get("head", {}).get("head_moved") for e in res["positions"].values()
    )

    parts = []
    for key, entry in res["positions"].items():
        name = key.replace("_", " ")
        if np.isfinite(entry["interocular_difference"]):
            parts.append(
                f"On {name}, the two eyes differed in measured excursion by "
                f"{entry['interocular_difference']:.2f} units."
            )
    if res["head_contaminated"]:
        parts.append("The head moved during at least one position, which corrupts these values.")
    res["description"] = " ".join(parts) or "Not enough usable frames to describe."
    return res


# ---------------------------------------------------------------------------
# 6. Fatigable saccades
# ---------------------------------------------------------------------------

def analyse_fatigable_saccades(phases: dict[str, list[FrameMeasure]]) -> dict:
    frames = phases.get("alternating", [])
    res = {"per_eye": {}}
    quality = _quality_block(frames, 45.0)

    for eye in ("right", "left"):
        ts, ys, blinks = [], [], []
        for f in frames:
            if not f.valid:
                continue
            e = getattr(f, eye)
            ts.append(f.t)
            ys.append(e.iris_x_mm)
            blinks.append(e.blinking)
        t = np.asarray(ts, float)
        y = np.asarray(ys, float)
        detail = {"n_saccades": 0}

        if t.size > 30:
            span = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
            vel_threshold = max(span * 2.0, 8.0)
            sacs = signals.detect_saccades(
                t, y,
                velocity_threshold=vel_threshold,
                min_amplitude=max(span * 0.3, 0.6),
                blink_mask=np.asarray(blinks, bool),
            )
            detail["n_saccades"] = len(sacs)
            if len(sacs) >= 6:
                st = np.asarray([s.t_start for s in sacs], float)
                amp = np.asarray([abs(s.amplitude) for s in sacs], float)
                vel = np.asarray([s.peak_velocity for s in sacs], float)
                amp_trend = signals.trend(st, amp, window_s=10.0)
                vel_trend = signals.trend(st, vel, window_s=10.0)
                detail.update(
                    {
                        "first_third_amplitude_mm": float(np.mean(amp[: len(amp) // 3])),
                        "last_third_amplitude_mm": float(np.mean(amp[-len(amp) // 3 :])),
                        "amplitude_decrement_pct": float(
                            100.0 * (1 - np.mean(amp[-len(amp) // 3 :]) /
                                     max(np.mean(amp[: len(amp) // 3]), 1e-6))
                        ),
                        "amplitude_slope_per_min": amp_trend.slope_per_min,
                        "velocity_slope_per_min": vel_trend.slope_per_min,
                        "amplitude_cov": signals.coefficient_of_variation(amp),
                    }
                )
        res["per_eye"][eye] = detail

    res["quality"] = quality
    res["velocity_meaningful"] = bool(
        np.isfinite(quality["fps"]) and quality["fps"] >= 100
    )
    parts = []
    for eye, label in (("right", "right"), ("left", "left")):
        d = res["per_eye"][eye]
        if "amplitude_decrement_pct" in d:
            parts.append(
                f"Across {d['n_saccades']} detected {label}-eye movements, the "
                f"average size of the last third was "
                f"{d['amplitude_decrement_pct']:.0f}% smaller than the first third."
            )
    parts.append(
        "Saccade speed cannot be measured at this camera's frame rate; only "
        "the size of the movements is shown."
        if not res["velocity_meaningful"] else ""
    )
    res["description"] = " ".join(p for p in parts if p) or "Not enough usable frames."
    return res


# ---------------------------------------------------------------------------
# 7. Gaze-holding instability
# ---------------------------------------------------------------------------

def analyse_gaze_holding(phases: dict[str, list[FrameMeasure]]) -> dict:
    res = {"holds": {}}
    for key in ("hold_right", "hold_left", "hold_up"):
        frames = phases.get(key, [])
        if not frames:
            continue
        attr = "gaze_v" if key == "hold_up" else "gaze_h"
        entry = {}
        for eye in ("right", "left"):
            t, y = _series(frames, eye, attr)
            d = signals.analyse_hold(t, y)
            entry[eye] = {
                "drift_rate_per_s": d.drift_rate,
                "rms_instability": d.rms_instability,
                "dominant_freq_hz": d.dominant_freq_hz,
                "oscillation_power": d.oscillation_power,
                "n": d.n,
            }
        entry["head"] = _head_moved(frames, limit_deg=5.0)
        entry["quality"] = _quality_block(frames, 20.0)
        res["holds"][key] = entry

    parts = []
    for key, entry in res["holds"].items():
        direction = key.replace("hold_", "")
        for eye in ("right", "left"):
            d = entry[eye]
            if np.isfinite(d["drift_rate_per_s"]) and abs(d["drift_rate_per_s"]) > 0.005:
                parts.append(
                    f"Holding gaze {direction}, the {eye} eye moved back towards "
                    f"centre at {abs(d['drift_rate_per_s']):.3f} units per second."
                )
    res["description"] = " ".join(parts) or "No appreciable drift was measured."
    return res


# ---------------------------------------------------------------------------
# 8. Head tilt / turn compensation
# ---------------------------------------------------------------------------

def analyse_head_posture(phases: dict[str, list[FrameMeasure]]) -> dict:
    neutral = phases.get("neutral_calibration", [])
    natural = phases.get("natural_posture", [])
    res = {}

    def _median(frames, attr):
        v = np.asarray([getattr(f, attr) for f in frames if f.valid], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    for attr, label in (
        ("head_roll_deg", "roll"),
        ("head_yaw_deg", "yaw"),
        ("head_pitch_deg", "pitch"),
    ):
        base = _median(neutral, attr)
        nat = _median(natural, attr)
        vals = np.asarray([getattr(f, attr) for f in natural if f.valid], float)
        vals = vals[np.isfinite(vals)] - (base if np.isfinite(base) else 0.0)
        offset = nat - base if np.isfinite(base) and np.isfinite(nat) else float("nan")
        if np.isfinite(offset):
            offset = float((offset + 180.0) % 360.0 - 180.0)
        res[label] = {
            "neutral_deg": base,
            "natural_deg": nat,
            "offset_deg": offset,
            "sd_deg": float(np.std(vals)) if vals.size > 2 else float("nan"),
            "pct_time_beyond_5deg": (
                float(100.0 * np.mean(np.abs(vals) > 5.0)) if vals.size else float("nan")
            ),
        }

    res["quality"] = _quality_block(natural, 10.0)
    parts = []
    for label, human in (("roll", "tilted"), ("yaw", "turned")):
        d = res[label]
        if np.isfinite(d["offset_deg"]) and abs(d["offset_deg"]) >= 2.0:
            side = "left" if d["offset_deg"] > 0 else "right"
            parts.append(
                f"During relaxed viewing the head was {human} {abs(d['offset_deg']):.0f} "
                f"degrees to the {side} compared with the deliberately straight position, "
                f"for {d['pct_time_beyond_5deg']:.0f}% of the time beyond 5 degrees."
            )
    res["description"] = " ".join(parts) or (
        "Head position during relaxed viewing was close to the straight reference."
    )
    return res


# ---------------------------------------------------------------------------
# 9. Variability (computed across repeats, not from its own recording)
# ---------------------------------------------------------------------------

def analyse_variability(phases: dict[str, list[FrameMeasure]]) -> dict:
    """Intra-exam variability of resting lid position across short samples."""
    samples = [v for k, v in phases.items() if k.startswith("sample_")]
    res = {"per_eye": {}, "n_samples": len(samples)}
    for eye in ("right", "left"):
        values = []
        for frames in samples:
            _, y = _series(frames, eye, "mrd1_mm")
            if y.size:
                values.append(float(np.median(y)))
        res["per_eye"][eye] = {
            "sample_values_mm": values,
            **signals.summarise(np.asarray(values, float)),
            "cov": signals.coefficient_of_variation(values),
        }
    parts = []
    for eye in ("right", "left"):
        d = res["per_eye"][eye]
        if d["n"] >= 2:
            parts.append(
                f"Across {d['n']} separate measurements a few minutes apart, the "
                f"{eye} upper lid position ranged over {d['range']:.1f} mm "
                f"(mean {d['mean']:.1f} mm)."
            )
    res["description"] = " ".join(parts) or "Not enough samples to describe."
    return res


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

PROTOCOLS: list[Protocol] = [
    Protocol(
        key="fatigable_ptosis",
        title="Fatigable ptosis",
        plain_title="Does an eyelid droop when you keep looking up?",
        why="Measures the height of each upper eyelid while you hold your eyes up for one minute.",
        needs_helper=False,
        caveat_key="fatigable_ptosis",
        phases=[
            Phase("baseline", "Look straight at the dot. Keep your head still and blink normally.", 8, CENTRE),
            Phase("upgaze", "Now look up at the dot near the top and KEEP looking at it. Do not move your head. Blinking is fine.", 60, UP),
            Phase("recovery", "Relax. Look straight ahead again and rest your eyes.", 20, CENTRE),
        ],
    ),
    Protocol(
        key="cogan_lid_twitch",
        title="Cogan's lid twitch",
        plain_title="Does an eyelid flick upward after looking down?",
        why="Looks for a brief upward overshoot of the eyelid when you look back up after looking down.",
        needs_helper=False,
        caveat_key="cogan_lid_twitch",
        min_fps=45.0,
        repeats=3,
        phases=[
            Phase("settle", "Look straight at the dot and get comfortable.", 5, CENTRE),
            Phase("downgaze", "Look down at the dot near the bottom. Hold it. Try not to blink.", 15, DOWN),
            Phase("refixate", "NOW look straight ahead quickly. Hold still and do not blink.", 4, CENTRE),
        ],
    ),
    Protocol(
        key="curtain_sign",
        title="Curtain sign (enhanced ptosis)",
        plain_title="Does one eyelid drop when the other is held up?",
        why="Measures one eyelid while a helper gently holds the other one open.",
        needs_helper=True,
        caveat_key="curtain_sign",
        phases=[
            Phase("baseline", "Both of you: nothing to do yet. Just look at the dot, head still.", 12, CENTRE),
            Phase(
                "lifted",
                "Keep looking at the dot. Stay relaxed. Do not squint.",
                15,
                CENTRE,
                helper_action=(
                    "HELPER: with one clean fingertip on the EYEBROW, gently lift the "
                    "droopier upper eyelid. Rest on the bone above the eye. "
                    "NEVER press on the eyeball itself. Stop at once if it hurts."
                ),
            ),
            Phase("release", "Helper, let go now. Keep looking at the dot.", 10, CENTRE),
        ],
    ),
    Protocol(
        key="peek_sign",
        title="Peek sign",
        plain_title="Do the eyes drift open when you close them gently?",
        why="Watches whether gently closed eyelids slowly part again.",
        needs_helper=False,
        caveat_key="peek_sign",
        phases=[
            Phase("baseline", "Look at the dot with your eyes open and relaxed.", 6, CENTRE),
            Phase(
                "gentle_closure",
                "Close your eyes GENTLY, like falling asleep. Do not squeeze. Keep them closed and stay relaxed.",
                45,
                None,
            ),
            Phase("open", "Open your eyes and look at the dot again.", 6, CENTRE),
        ],
    ),
    Protocol(
        key="ophthalmoparesis",
        title="Variable / asymmetric ophthalmoparesis",
        plain_title="Do both eyes move the same amount in each direction?",
        why="Compares how far each eye travels when you look in different directions.",
        needs_helper=False,
        caveat_key="ophthalmoparesis",
        repeats=3,
        phases=[
            Phase("centre", "Look at the dot in the middle. Head still.", 4, CENTRE),
            Phase("right_gaze", "Follow the dot to the right. Move only your eyes.", 4, RIGHT),
            Phase("centre_2", "Back to the middle.", 3, CENTRE),
            Phase("left_gaze", "Follow the dot to the left. Move only your eyes.", 4, LEFT),
            Phase("centre_3", "Back to the middle.", 3, CENTRE),
            Phase("up_gaze", "Follow the dot upward. Move only your eyes.", 4, UP),
            Phase("centre_4", "Back to the middle.", 3, CENTRE),
            Phase("down_gaze", "Follow the dot downward. Move only your eyes.", 4, DOWN),
        ],
    ),
    Protocol(
        key="fatigable_saccades",
        title="Fatigable saccades",
        plain_title="Do quick eye movements get smaller as you repeat them?",
        why="Measures the size of rapid left-right eye movements over one minute.",
        needs_helper=False,
        caveat_key="fatigable_saccades",
        min_fps=45.0,
        phases=[
            Phase("centre", "Look at the middle dot.", 4, CENTRE),
            Phase(
                "alternating",
                "The dot will jump left and right. Snap your eyes to it each time. "
                "Keep your head completely still. Keep going even if it feels tiring.",
                60,
                None,
            ),
            Phase("rest", "Relax and look at the middle dot.", 10, CENTRE),
        ],
    ),
    Protocol(
        key="gaze_holding",
        title="Gaze-holding instability",
        plain_title="Can your eyes stay still when looking to the side?",
        why="Measures whether the eyes drift back towards the middle when held to one side.",
        needs_helper=False,
        caveat_key="gaze_holding",
        min_fps=20.0,
        phases=[
            Phase("hold_right", "Look at the dot on the right and hold it as steady as you can.", 20, RIGHT),
            Phase("rest_1", "Back to the middle. Rest.", 6, CENTRE),
            Phase("hold_left", "Look at the dot on the left and hold it steady.", 20, LEFT),
            Phase("rest_2", "Back to the middle. Rest.", 6, CENTRE),
            Phase("hold_up", "Look at the dot at the top and hold it steady.", 20, UP),
        ],
    ),
    Protocol(
        key="head_posture",
        title="Head tilt / turn compensation",
        plain_title="Do you hold your head at an angle without meaning to?",
        why="Compares your relaxed head position with a deliberately straight one.",
        needs_helper=False,
        caveat_key="head_posture",
        phases=[
            Phase(
                "neutral_calibration",
                "Deliberately hold your head perfectly straight and level, facing the camera. "
                "Look at the dot.",
                12,
                CENTRE,
            ),
            Phase(
                "natural_posture",
                "Now stop thinking about your head. Just read the dot comfortably, "
                "however feels natural. Do not correct yourself.",
                30,
                CENTRE,
            ),
        ],
    ),
    Protocol(
        key="variability",
        title="Variability within this exam",
        plain_title="How much does the measurement change from minute to minute?",
        why="Takes several short measurements so you can see how much the numbers wander on their own.",
        needs_helper=False,
        caveat_key="variability",
        phases=[
            Phase("sample_1", "Look at the dot, relaxed, normal blinking.", 10, CENTRE),
            Phase("pause_1", "Close your eyes and rest for a moment.", 20, None, record=False),
            Phase("sample_2", "Look at the dot again, relaxed.", 10, CENTRE),
            Phase("pause_2", "Rest your eyes again.", 20, None, record=False),
            Phase("sample_3", "Last one. Look at the dot, relaxed.", 10, CENTRE),
        ],
    ),
]

ANALYSERS = {
    "fatigable_ptosis": analyse_fatigable_ptosis,
    "cogan_lid_twitch": analyse_cogan,
    "curtain_sign": analyse_curtain,
    "peek_sign": analyse_peek,
    "ophthalmoparesis": analyse_ophthalmoparesis,
    "fatigable_saccades": analyse_fatigable_saccades,
    "gaze_holding": analyse_gaze_holding,
    "head_posture": analyse_head_posture,
    "variability": analyse_variability,
}

for _p in PROTOCOLS:
    _p.analyse = ANALYSERS[_p.key]


def get_protocol(key: str) -> Protocol:
    for p in PROTOCOLS:
        if p.key == key:
            return p
    raise KeyError(key)


def caveat_for(key: str) -> str:
    return TEST_CAVEATS.get(key, "")
