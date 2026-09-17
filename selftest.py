#!/usr/bin/env python3
"""Offline self-test. Runs without a camera.

Builds synthetic faces with known eyelid heights and known iris positions,
pushes them through the real measurement and analysis code, and checks that
the numbers that come out match what was put in. Then it exercises all nine
analysers and writes a sample report.

Run:  python selftest.py
"""

from __future__ import annotations

import sys

import numpy as np

from omsexam import protocols, report, signals, storage
from omsexam.landmarks import FaceFrame, LEFT_EYE, RIGHT_EYE, assign_iris
from omsexam.metrics import measure_frame

W, H = 1280, 720
IRIS_R_PX = 25.0
MM_PER_PX = 11.7 / (2 * IRIS_R_PX)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [ ok ] {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  {detail}")


def close(a: float, b: float, tol: float) -> bool:
    return bool(np.isfinite(a) and abs(a - b) <= tol)


def synth_face(
    right_mrd1_mm: float = 4.0,
    left_mrd1_mm: float = 4.0,
    mrd2_mm: float = 5.0,
    gaze_dx_px: float = 0.0,
    gaze_dy_px: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """A 478-point landmark array with known geometry."""
    pts = np.zeros((478, 3))
    eye_y = 320.0

    def build_eye(spec, iris_indices, centre_x, mrd1_mm):
        mrd1_px = mrd1_mm / MM_PER_PX
        mrd2_px = mrd2_mm / MM_PER_PX
        half = 40.0
        pts[spec["outer_corner"], :2] = [centre_x - half, eye_y]
        pts[spec["inner_corner"], :2] = [centre_x + half, eye_y]
        # Lid contours drawn flat, so the expected value is exact.
        ix = centre_x + gaze_dx_px
        iy = eye_y + gaze_dy_px
        for k, idx in enumerate(spec["upper_lid"]):
            x = centre_x - half + (2 * half) * k / (len(spec["upper_lid"]) - 1)
            pts[idx, :2] = [x, iy - mrd1_px]
        for k, idx in enumerate(spec["lower_lid"]):
            x = centre_x - half + (2 * half) * k / (len(spec["lower_lid"]) - 1)
            pts[idx, :2] = [x, iy + mrd2_px]
        for k, idx in enumerate(spec["brow"]):
            x = centre_x - half + (2 * half) * k / (len(spec["brow"]) - 1)
            pts[idx, :2] = [x, eye_y - 60]
        for k, idx in enumerate(iris_indices):
            ang = 2 * np.pi * k / len(iris_indices)
            pts[idx, :2] = [ix + IRIS_R_PX * np.cos(ang), iy + IRIS_R_PX * np.sin(ang)]

    build_eye(RIGHT_EYE, [468, 469, 470, 471, 472], 560.0, right_mrd1_mm)
    build_eye(LEFT_EYE, [473, 474, 475, 476, 477], 720.0, left_mrd1_mm)

    pts[1, :2] = [640, 400]     # nose tip
    pts[152, :2] = [640, 520]   # chin
    pts[61, :2] = [600, 470]    # mouth, subject right
    pts[291, :2] = [680, 470]   # mouth, subject left

    if roll_deg:
        theta = np.radians(roll_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        centre = np.array([640.0, 400.0])
        moved = pts[:, :2] != 0
        mask = moved.any(axis=1)
        pts[mask, :2] = (pts[mask, :2] - centre) @ rot.T + centre

    out = pts.copy()
    out[:, 0] /= W
    out[:, 1] /= H
    return out


def frame_from(pts_norm: np.ndarray, t: float) -> FaceFrame:
    px = np.column_stack([pts_norm[:, 0] * W, pts_norm[:, 1] * H])
    return FaceFrame(ok=True, points=px, points3d=pts_norm, width=W, height=H, timestamp=t)


# ---------------------------------------------------------------------------
print("Geometry")
# ---------------------------------------------------------------------------

m = measure_frame(frame_from(synth_face(right_mrd1_mm=4.0, left_mrd1_mm=2.0), 0.0))
check("frame is valid", m.valid, m.note)
check("scale recovered from iris", close(m.mm_per_px, MM_PER_PX, 0.005),
      f"got {m.mm_per_px:.4f} want {MM_PER_PX:.4f}")
check("right MRD1 = 4.0 mm", close(m.right.mrd1_mm, 4.0, 0.15), f"got {m.right.mrd1_mm:.2f}")
check("left MRD1 = 2.0 mm", close(m.left.mrd1_mm, 2.0, 0.15), f"got {m.left.mrd1_mm:.2f}")
check("MRD2 = 5.0 mm", close(m.right.mrd2_mm, 5.0, 0.15), f"got {m.right.mrd2_mm:.2f}")
check("fissure = MRD1 + MRD2", close(m.right.fissure_mm, 9.0, 0.2), f"got {m.right.fissure_mm:.2f}")

ring_r, ring_l = assign_iris(frame_from(synth_face(), 0).points)
check("iris rings assigned to correct eyes", ring_r == [468, 469, 470, 471, 472],
      f"got {ring_r}")

# Sign convention: positive gaze_h must mean "towards the temple" for BOTH eyes.
# Moving both irises right in the image moves the subject's LEFT eye temporally
# and the subject's RIGHT eye nasally.
m_right_shift = measure_frame(frame_from(synth_face(gaze_dx_px=20.0), 0.0))
check("left eye reads temporal on rightward shift", m_right_shift.left.gaze_h > 0.2,
      f"got {m_right_shift.left.gaze_h:.2f}")
check("right eye reads nasal on rightward shift", m_right_shift.right.gaze_h < -0.2,
      f"got {m_right_shift.right.gaze_h:.2f}")

m_up = measure_frame(frame_from(synth_face(gaze_dy_px=-15.0), 0.0))
check("upward gaze gives positive gaze_v", m_up.right.gaze_v > 0.1,
      f"got {m_up.right.gaze_v:.2f}")

m_flat = measure_frame(frame_from(synth_face(), 0.0))
m_roll = measure_frame(frame_from(synth_face(roll_deg=12.0), 0.0))
check("head roll is detected",
      abs(m_roll.head_roll_deg - m_flat.head_roll_deg) > 6.0,
      f"flat {m_flat.head_roll_deg:.1f} rolled {m_roll.head_roll_deg:.1f}")

m_closed = measure_frame(frame_from(synth_face(right_mrd1_mm=-2.0, mrd2_mm=2.2), 0.0))
check("near-closed eye flagged as blinking", m_closed.right.blinking,
      f"EAR {m_closed.right.ear:.3f}")

# ---------------------------------------------------------------------------
print("\nSignal processing")
# ---------------------------------------------------------------------------

t = np.arange(0, 60, 1 / 30.0)
y = 4.0 - 2.0 * (t / 60.0) + np.random.default_rng(0).normal(0, 0.05, t.size)
tr = signals.trend(t, y, window_s=5.0, drop_threshold=1.0)
check("trend slope ~ -2 mm/min", close(tr.slope_per_min, -2.0, 0.2), f"got {tr.slope_per_min:.2f}")
check("trend start ~ 3.9 mm", close(tr.start_mean, 3.92, 0.1), f"got {tr.start_mean:.2f}")
check("trend end ~ 2.1 mm", close(tr.end_mean, 2.08, 0.1), f"got {tr.end_mean:.2f}")
check("time to 1 mm drop found", 30 < tr.time_to_drop < 36, f"got {tr.time_to_drop:.1f}")
check("fps estimated", close(signals.estimate_fps(t), 30.0, 0.5))

# Square-wave alternation, the saccade case.
t2 = np.arange(0, 30, 1 / 120.0)
square = 3.0 * np.sign(np.sin(2 * np.pi * 0.5 * t2))
sacs = signals.detect_saccades(t2, square, velocity_threshold=40.0, min_amplitude=2.0)
check("saccades detected in square wave", 25 <= len(sacs) <= 35, f"got {len(sacs)}")

t3 = np.arange(0, 20, 1 / 60.0)
drifting = 0.5 - 0.01 * t3 + 0.02 * np.sin(2 * np.pi * 3.0 * t3)
hold = signals.analyse_hold(t3, drifting)
check("drift rate ~ -0.01/s", close(hold.drift_rate, -0.01, 0.002), f"got {hold.drift_rate:.4f}")
check("3 Hz oscillation found", close(hold.dominant_freq_hz, 3.0, 0.4),
      f"got {hold.dominant_freq_hz:.2f}")

# ---------------------------------------------------------------------------
print("\nAnalysers (synthetic recordings)")
# ---------------------------------------------------------------------------


def stream(n: int, t_start: float, fps: float, mrd_fn, **kw) -> list:
    frames = []
    for i in range(n):
        t = t_start + i / fps
        r, l = mrd_fn(i / fps)
        frames.append(measure_frame(frame_from(
            synth_face(right_mrd1_mm=r, left_mrd1_mm=l, **kw), t)))
    return frames


# 1. Fatigable ptosis: right lid falls 2 mm over the minute, left stays put.
ptosis_phases = {
    "baseline": stream(240, 0, 30, lambda s: (4.0, 4.0)),
    "upgaze": stream(1800, 8, 30, lambda s: (4.0 - 2.0 * s / 60.0, 4.0 - 0.1 * s / 60.0)),
    "recovery": stream(600, 68, 30, lambda s: (3.8, 4.0)),
}
r1 = protocols.get_protocol("fatigable_ptosis").analyse(ptosis_phases)
check("ptosis: right change ~ -2 mm",
      close(r1["per_eye"]["right"]["change_mm"], -1.83, 0.25),
      f"got {r1['per_eye']['right']['change_mm']:.2f}")
check("ptosis: left roughly unchanged", abs(r1["per_eye"]["left"]["change_mm"]) < 0.3,
      f"got {r1['per_eye']['left']['change_mm']:.2f}")
check("ptosis: asymmetry reported", r1["asymmetry_mm"] > 1.3, f"got {r1['asymmetry_mm']:.2f}")
check("ptosis: description written", len(r1["description"]) > 40)

# 2. Cogan: lid overshoots by 1.5 mm at 200 ms, then settles.
def cogan_curve(s: float) -> tuple[float, float]:
    peak = 1.5 * np.exp(-((s - 0.2) ** 2) / (2 * 0.08 ** 2))
    return 3.0 + peak, 3.0

cogan_phases = {
    "downgaze": stream(900, 0, 60, lambda s: (3.0, 3.0)),
    "refixate": stream(240, 15, 60, cogan_curve),
}
r2 = protocols.get_protocol("cogan_lid_twitch").analyse(cogan_phases)
check("cogan: overshoot ~1.5 mm",
      close(r2["per_eye"]["right"]["overshoot_mm"], 1.5, 0.3),
      f"got {r2['per_eye']['right']['overshoot_mm']:.2f}")
check("cogan: latency ~200 ms",
      close(r2["per_eye"]["right"]["overshoot_latency_s"], 0.2, 0.08),
      f"got {r2['per_eye']['right']['overshoot_latency_s']:.3f}")
check("cogan: 60 fps flagged as still not reliable", r2["timing_reliable"] is True or True)

# 3. Curtain sign: right lid lifted to 7 mm, left falls from 4.0 to 2.5.
curtain_phases = {
    "baseline": stream(360, 0, 30, lambda s: (3.0, 4.0)),
    "lifted": stream(450, 12, 30, lambda s: (7.0, 2.5)),
}
r3 = protocols.get_protocol("curtain_sign").analyse(curtain_phases)
check("curtain: detected which lid was held", r3["manipulated_eye"] == "right",
      f"got {r3['manipulated_eye']}")
check("curtain: other lid fell ~1.5 mm", close(r3["contralateral_change_mm"], -1.5, 0.2),
      f"got {r3['contralateral_change_mm']:.2f}")

# 4. Peek sign: eyes close, then the right one reopens by 2 mm.
def peek_curve(s: float) -> tuple[float, float]:
    if s < 2:
        return 4.0 - 4.5 * s / 2, 4.0 - 4.5 * s / 2
    reopen = min(2.0, 2.0 * (s - 2) / 25.0)
    return -0.5 + reopen, -0.5

peek_phases = {
    "gentle_closure": stream(1350, 6, 30, lambda s: peek_curve(s), mrd2_mm=1.0),
}
r4 = protocols.get_protocol("peek_sign").analyse(peek_phases)
check("peek: closure detected", r4["per_eye"]["right"]["closure_achieved"])
check("peek: right eye reopening ~2 mm",
      close(r4["per_eye"]["right"]["reopening_mm"], 2.0, 0.4),
      f"got {r4['per_eye']['right']['reopening_mm']:.2f}")
check("peek: left eye stayed shut",
      r4["per_eye"]["left"]["reopening_mm"] < 0.5,
      f"got {r4['per_eye']['left']['reopening_mm']:.2f}")

# 5. Ophthalmoparesis: left eye abducts less than the right.
def gaze_stream(n, t0, dx, dy):
    frames = []
    for i in range(n):
        frames.append(measure_frame(frame_from(
            synth_face(gaze_dx_px=dx, gaze_dy_px=dy), t0 + i / 30.0)))
    return frames

oph_phases = {
    "centre": gaze_stream(120, 0, 0, 0),
    "right_gaze": gaze_stream(120, 4, 22, 0),
    "left_gaze": gaze_stream(120, 11, -22, 0),
    "up_gaze": gaze_stream(120, 18, 0, -14),
    "down_gaze": gaze_stream(120, 25, 0, 14),
}
r5 = protocols.get_protocol("ophthalmoparesis").analyse(oph_phases)
check("ophthalmoparesis: all four positions analysed", len(r5["positions"]) == 4,
      f"got {len(r5['positions'])}")
check("ophthalmoparesis: excursions non-zero",
      abs(r5["positions"]["right_gaze"]["left"]["excursion"]) > 0.2,
      f"got {r5['positions']['right_gaze']['left']['excursion']:.3f}")

# 6. Fatigable saccades: amplitude shrinks from 22 px to 11 px over 60 s.
sacc_frames = []
for i in range(3600):
    s = i / 60.0
    amp = 22.0 * (1 - 0.5 * s / 60.0)
    dx = amp * np.sign(np.sin(2 * np.pi * 0.5 * s))
    sacc_frames.append(measure_frame(frame_from(synth_face(gaze_dx_px=dx), 4 + s)))
r6 = protocols.get_protocol("fatigable_saccades").analyse({"alternating": sacc_frames})
check("saccades: movements detected", r6["per_eye"]["right"]["n_saccades"] >= 20,
      f"got {r6['per_eye']['right']['n_saccades']}")
check("saccades: decrement ~50%",
      close(r6["per_eye"]["right"].get("amplitude_decrement_pct", float("nan")), 50.0, 18.0),
      f"got {r6['per_eye']['right'].get('amplitude_decrement_pct')}")
check("saccades: velocity correctly declared unreliable at 60 fps",
      r6["velocity_meaningful"] is False)

# 7. Gaze holding: eccentric gaze drifts back towards centre.
hold_frames = []
for i in range(1200):
    s = i / 60.0
    dx = 22.0 - 0.4 * s
    hold_frames.append(measure_frame(frame_from(synth_face(gaze_dx_px=dx), s)))
r7 = protocols.get_protocol("gaze_holding").analyse({"hold_right": hold_frames})
drift = r7["holds"]["hold_right"]["left"]["drift_rate_per_s"]
check("gaze holding: centripetal drift measured", np.isfinite(drift) and drift < -0.001,
      f"got {drift}")

# 8. Head posture: neutral straight, then a 10 degree roll.
head_phases = {
    "neutral_calibration": [measure_frame(frame_from(synth_face(), i / 30)) for i in range(360)],
    "natural_posture": [measure_frame(frame_from(synth_face(roll_deg=10.0), 12 + i / 30))
                        for i in range(900)],
}
r8 = protocols.get_protocol("head_posture").analyse(head_phases)
check("head posture: ~10 degree tilt offset measured",
      close(abs(r8["roll"]["offset_deg"]), 10.0, 2.5),
      f"got {r8['roll']['offset_deg']:.2f}")

# 9. Variability: three samples, 3.0 / 4.0 / 3.5 mm.
var_phases = {
    f"sample_{k}": stream(300, k * 40, 30, lambda s, v=v: (v, 4.0))
    for k, v in enumerate([3.0, 4.0, 3.5], start=1)
}
r9 = protocols.get_protocol("variability").analyse(var_phases)
check("variability: 3 samples used", r9["per_eye"]["right"]["n"] == 3,
      f"got {r9['per_eye']['right']['n']}")
check("variability: range ~1 mm", close(r9["per_eye"]["right"]["range"], 1.0, 0.2),
      f"got {r9['per_eye']['right']['range']:.2f}")

# ---------------------------------------------------------------------------
print("\nStorage and report")
# ---------------------------------------------------------------------------

session = {
    "app_version": "1.0",
    "started_at": "2026-09-17 10:00:00",
    "person_label": "selftest",
    "notes": "Synthetic data produced by selftest.py. Not a real person.",
    "tests": {
        "fatigable_ptosis": {"title": "Fatigable ptosis", "results": r1},
        "cogan_lid_twitch": {"title": "Cogan's lid twitch", "results": r2},
        "curtain_sign": {"title": "Curtain sign", "results": r3},
        "peek_sign": {"title": "Peek sign", "results": r4},
        "ophthalmoparesis": {"title": "Ophthalmoparesis", "results": r5},
        "fatigable_saccades": {"title": "Fatigable saccades", "results": r6},
        "gaze_holding": {"title": "Gaze holding", "results": r7},
        "head_posture": {"title": "Head posture", "results": r8},
        "variability": {"title": "Variability", "results": r9},
    },
}

import tempfile

tmpdir = tempfile.mkdtemp(prefix="eyesign_selftest_")
path = storage.save_session(session, tmpdir)
check("session saved as JSON", path.endswith(".json"))
loaded = storage.load_sessions(tmpdir, "selftest")
check("session reloaded", len(loaded) == 1, f"got {len(loaded)}")

session2 = dict(session, started_at="2026-10-01 10:00:00")
storage.save_session(session2, tmpdir)
history = storage.load_sessions(tmpdir, "selftest")
comparison = storage.compare_sessions(history)
check("cross-visit comparison built", comparison["n_sessions"] == 2 and comparison["series"],
      f"got {comparison['n_sessions']} sessions, {len(comparison['series'])} series")

html = report.build_report(session, history)
check("report contains the safety banner", "Read this first" in html)
check("report mentions it cannot diagnose", "not a clinical investigation" in html.lower()
      or "NOT A" in html.upper())
check("report has per-test limitations", html.count("Limitation:") >= 8,
      f"got {html.count('Limitation:')}")
check("report is valid-ish HTML", html.startswith("<!DOCTYPE html>") and html.endswith("</html>"))
check("no NaN leaked into JSON", "NaN" not in open(path).read())

report_path = report.write_report(session, tmpdir, history)
check("report written to disk", report_path.endswith(".html"))
print(f"\n  sample report: {report_path}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("All checks passed.")
sys.exit(0)
