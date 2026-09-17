"""Signal processing on the per-frame measurement streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clean(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN samples, keeping the two arrays aligned."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    return t[m], y[m]


def median_filter(y: np.ndarray, k: int = 5) -> np.ndarray:
    """Odd-length running median. Removes single-frame tracking spikes."""
    y = np.asarray(y, dtype=float)
    if k < 3 or y.size < k:
        return y.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(y, pad, mode="edge")
    out = np.empty_like(y)
    for i in range(y.size):
        out[i] = np.median(padded[i : i + k])
    return out


def estimate_fps(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 3:
        return float("nan")
    dt = np.diff(t)
    dt = dt[(dt > 1e-4) & (dt < 1.0)]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


@dataclass
class Trend:
    slope_per_min: float = float("nan")
    intercept: float = float("nan")
    r2: float = float("nan")
    start_mean: float = float("nan")
    end_mean: float = float("nan")
    change: float = float("nan")          # end_mean - start_mean
    n: int = 0
    duration_s: float = float("nan")
    time_to_drop: float = float("nan")    # s until sustained drop past a limit


def trend(
    t: np.ndarray,
    y: np.ndarray,
    window_s: float = 5.0,
    drop_threshold: float = 1.0,
    sustain_s: float = 1.0,
) -> Trend:
    """Linear trend plus start/end window comparison.

    `time_to_drop` is the first moment after which the signal stays at least
    `drop_threshold` below its opening baseline for `sustain_s` seconds.
    """
    t, y = clean(t, y)
    out = Trend(n=int(t.size))
    if t.size < 8:
        return out
    t0 = t - t[0]
    out.duration_s = float(t0[-1])

    a_mat = np.column_stack([t0, np.ones_like(t0)])
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    slope_per_s, intercept = float(coef[0]), float(coef[1])
    out.slope_per_min = slope_per_s * 60.0
    out.intercept = intercept

    pred = a_mat @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    out.r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    head = y[t0 <= window_s]
    tail = y[t0 >= max(out.duration_s - window_s, 0.0)]
    if head.size:
        out.start_mean = float(np.mean(head))
    if tail.size:
        out.end_mean = float(np.mean(tail))
    if head.size and tail.size:
        out.change = out.end_mean - out.start_mean

    if np.isfinite(out.start_mean):
        # Smooth first: without this, noise straddling the threshold keeps
        # resetting the sustain timer and the reported time drifts late.
        smooth = median_filter(y, 9)
        below = smooth < (out.start_mean - drop_threshold)
        run_start = None
        for i, flag in enumerate(below):
            if flag:
                if run_start is None:
                    run_start = t0[i]
                elif t0[i] - run_start >= sustain_s:
                    out.time_to_drop = float(run_start)
                    break
            else:
                run_start = None
    return out


@dataclass
class Saccade:
    t_start: float = float("nan")
    t_end: float = float("nan")
    amplitude: float = float("nan")       # signed, same units as input
    peak_velocity: float = float("nan")   # units per second
    duration_s: float = float("nan")


def detect_saccades(
    t: np.ndarray,
    y: np.ndarray,
    velocity_threshold: float,
    min_amplitude: float,
    blink_mask: np.ndarray | None = None,
) -> list[Saccade]:
    """Velocity-threshold saccade detection on a position signal.

    Deliberately conservative. On a 30 fps camera a saccade lasts one to two
    frames, so this recovers amplitude reasonably and peak velocity badly.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if blink_mask is not None:
        y = y.copy()
        y[np.asarray(blink_mask, bool)] = np.nan
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 5:
        return []

    y = median_filter(y, 3)
    dt = np.diff(t)
    dt[dt <= 1e-6] = np.nan
    vel = np.diff(y) / dt
    vel = np.nan_to_num(vel, nan=0.0)
    speed = np.abs(vel)

    saccades: list[Saccade] = []
    i = 0
    while i < speed.size:
        if speed[i] < velocity_threshold:
            i += 1
            continue
        j = i
        while j < speed.size and speed[j] >= velocity_threshold * 0.4:
            j += 1
        i0, i1 = i, min(j, y.size - 1)
        amp = float(y[i1] - y[i0])
        if abs(amp) >= min_amplitude:
            saccades.append(
                Saccade(
                    t_start=float(t[i0]),
                    t_end=float(t[i1]),
                    amplitude=amp,
                    peak_velocity=float(np.max(speed[i0:max(i1, i0 + 1)])),
                    duration_s=float(t[i1] - t[i0]),
                )
            )
        i = j + 1
    return saccades


@dataclass
class DriftResult:
    drift_rate: float = float("nan")      # units per second, signed
    rms_instability: float = float("nan")
    dominant_freq_hz: float = float("nan")
    oscillation_power: float = float("nan")
    n: int = 0


def analyse_hold(
    t: np.ndarray,
    y: np.ndarray,
    fps_hint: float | None = None,
) -> DriftResult:
    """Drift and oscillation during an attempted steady eccentric fixation."""
    t, y = clean(t, y)
    out = DriftResult(n=int(t.size))
    if t.size < 16:
        return out

    t0 = t - t[0]
    y = median_filter(y, 3)
    coef, *_ = np.linalg.lstsq(np.column_stack([t0, np.ones_like(t0)]), y, rcond=None)
    out.drift_rate = float(coef[0])

    residual = y - (coef[0] * t0 + coef[1])
    out.rms_instability = float(np.sqrt(np.mean(residual ** 2)))

    fps = fps_hint or estimate_fps(t)
    if np.isfinite(fps) and fps > 4:
        # Resample onto a uniform grid so the FFT means something.
        grid = np.arange(t0[0], t0[-1], 1.0 / fps)
        if grid.size >= 32:
            resampled = np.interp(grid, t0, residual)
            resampled -= resampled.mean()
            window = np.hanning(resampled.size)
            spectrum = np.abs(np.fft.rfft(resampled * window)) ** 2
            freqs = np.fft.rfftfreq(resampled.size, 1.0 / fps)
            band = (freqs >= 0.8) & (freqs <= min(8.0, fps / 2.5))
            if band.any() and spectrum[band].size:
                k = int(np.argmax(spectrum[band]))
                out.dominant_freq_hz = float(freqs[band][k])
                total = float(np.sum(spectrum[1:])) or 1.0
                out.oscillation_power = float(spectrum[band][k] / total)
    return out


def coefficient_of_variation(values: list[float] | np.ndarray) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return float("nan")
    mean = float(np.mean(v))
    if abs(mean) < 1e-9:
        return float("nan")
    return float(np.std(v, ddof=1) / abs(mean))


def summarise(values: np.ndarray) -> dict:
    v = np.asarray([x for x in np.asarray(values, float) if np.isfinite(x)])
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "min": float("nan"), "max": float("nan"), "range": float("nan")}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "range": float(np.max(v) - np.min(v)),
    }
