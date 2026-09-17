"""Build a printable HTML report for a clinician.

Self-contained: inline CSS and hand-rolled SVG charts, so it opens in any
browser and prints cleanly without matplotlib or an internet connection.
"""

from __future__ import annotations

import html
import os
from datetime import datetime

from . import disclaimers, protocols, storage


def _fmt(value, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        try:
            if value != value:  # NaN
                return "&mdash;"
        except TypeError:
            return "&mdash;"
        return f"{value:.{digits}f}{unit}"
    return html.escape(str(value))


def _sparkline(values: list, width: int = 420, height: int = 90) -> str:
    pts = [v for v in values if isinstance(v, (int, float))]
    if len(pts) < 2:
        return "<p class='muted'>Not enough visits to draw a trend.</p>"
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    pad = 14
    step = (width - 2 * pad) / (len(values) - 1)

    coords, circles = [], []
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            continue
        x = pad + i * step
        y = height - pad - ((v - lo) / span) * (height - 2 * pad)
        coords.append(f"{x:.1f},{y:.1f}")
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#1a4d8f'>"
            f"<title>{v:.2f}</title></circle>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' "
        f"style='max-width:{width}px'>"
        f"<rect width='{width}' height='{height}' fill='#f7f9fc' stroke='#ccd6e4'/>"
        f"<polyline points='{' '.join(coords)}' fill='none' "
        f"stroke='#1a4d8f' stroke-width='2.5'/>"
        f"{''.join(circles)}"
        f"<text x='{pad}' y='12' font-size='11' fill='#667'>high {hi:.2f}</text>"
        f"<text x='{pad}' y='{height - 3}' font-size='11' fill='#667'>low {lo:.2f}</text>"
        "</svg>"
    )


def _quality_banner(quality: dict | None) -> str:
    if not quality:
        return ""
    problems = quality.get("problems") or []
    if quality.get("usable") and not problems:
        return (
            "<p class='ok'>Recording quality: acceptable "
            f"({_fmt(quality.get('fps'), 0)} fps, "
            f"{_fmt((quality.get('tracking_rate') or 0) * 100, 0)}% of frames tracked).</p>"
        )
    items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)
    return (
        "<div class='warn'><strong>Poor recording quality.</strong>"
        f"<ul>{items}</ul><p>{html.escape(disclaimers.LOW_CONFIDENCE_NOTE)}</p></div>"
    )


def _flatten(prefix: str, obj, rows: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("description", "problems"):
                continue
            _flatten(f"{prefix}{k}." if prefix else f"{k}.", v, rows) if isinstance(
                v, (dict, list)
            ) else rows.append((f"{prefix}{k}", v))
    elif isinstance(obj, list):
        rows.append((prefix.rstrip("."), ", ".join(_fmt(v) for v in obj)))


def _measurement_table(results: dict) -> str:
    rows: list = []
    for key, value in results.items():
        if key in ("description", "quality"):
            continue
        _flatten(f"{key}.", value, rows) if isinstance(value, (dict, list)) else rows.append(
            (key, value)
        )
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{_fmt(v)}</td></tr>"
        for k, v in rows
    )
    return (
        "<details><summary>All measured values</summary>"
        f"<table class='data'><thead><tr><th>Measurement</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table></details>"
    )


CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:0 auto;
padding:28px;color:#1a1a1a;line-height:1.55;font-size:17px}
h1{font-size:28px;margin-bottom:4px}
h2{font-size:22px;border-bottom:2px solid #1a4d8f;padding-bottom:6px;margin-top:38px}
h3{font-size:19px;margin-bottom:4px}
.banner{background:#fff4f4;border:3px solid #b32020;padding:18px;margin:22px 0;
border-radius:6px}
.banner h2{border:none;color:#b32020;margin-top:0;font-size:20px}
.warn{background:#fff8e6;border-left:6px solid #c88a00;padding:12px 16px;margin:12px 0}
.ok{color:#2a6a3a;font-size:15px}
.muted{color:#666;font-size:15px}
.caveat{background:#f2f4f8;border-left:6px solid #7a8aa0;padding:12px 16px;
margin:12px 0;font-size:15.5px}
.finding{background:#f7f9fc;border:1px solid #ccd6e4;padding:14px 18px;
border-radius:4px;margin:10px 0}
table.data{border-collapse:collapse;width:100%;font-size:14.5px;
font-family:'DejaVu Sans Mono',Menlo,monospace;margin-top:10px}
table.data th,table.data td{border:1px solid #d5dce6;padding:5px 9px;text-align:left}
table.data th{background:#eef2f7}
details{margin:10px 0}summary{cursor:pointer;color:#1a4d8f;font-size:15px}
footer{margin-top:44px;border-top:1px solid #ccc;padding-top:14px;
font-size:14px;color:#555}
@media print{.banner{border-width:2px}details{display:none}}
"""


def build_report(session: dict, history: list[dict] | None = None) -> str:
    started = session.get("started_at", "")
    person = html.escape(session.get("person_label", "(not named)"))
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Eye observation record - {person}</title><style>{CSS}</style></head><body>",
        "<h1>Eye observation record</h1>",
        f"<p class='muted'>Person: <strong>{person}</strong> &middot; "
        f"Recorded: {html.escape(started)} &middot; "
        f"Software: {disclaimers.APP_NAME} {disclaimers.APP_VERSION}</p>",
        "<div class='banner'><h2>Read this first</h2><pre style='white-space:pre-wrap;"
        "font-family:inherit;margin:0'>"
        + html.escape(disclaimers.REPORT_HEADER_WARNING)
        + "</pre></div>",
    ]

    if session.get("notes"):
        parts.append(
            f"<h2>Notes entered by the person</h2><p>{html.escape(session['notes'])}</p>"
        )

    parts.append("<h2>What was recorded</h2>")
    for test_key, test_data in (session.get("tests") or {}).items():
        try:
            proto = protocols.get_protocol(test_key)
            title, plain, why = proto.title, proto.plain_title, proto.why
        except KeyError:
            title, plain, why = test_key, "", ""
        results = test_data.get("results", {})
        parts.append(f"<h3>{html.escape(title)}</h3>")
        if plain:
            parts.append(f"<p class='muted'>{html.escape(plain)} {html.escape(why)}</p>")
        parts.append(_quality_banner(results.get("quality")))
        desc = results.get("description") or "No description available."
        parts.append(f"<div class='finding'>{html.escape(desc)}</div>")
        caveat = protocols.caveat_for(test_key)
        if caveat:
            parts.append(f"<div class='caveat'><strong>Limitation: </strong>{html.escape(caveat)}</div>")
        parts.append(_measurement_table(results))

    if history and len(history) > 1:
        parts.append("<h2>Comparison with earlier sessions</h2>")
        comparison = storage.compare_sessions(history)
        parts.append(f"<p class='caveat'>{html.escape(comparison['note'])}</p>")
        for entry in comparison["series"].values():
            summary = entry["summary"]
            parts.append(
                f"<h3>{html.escape(entry['label'])}</h3>"
                + _sparkline(entry["values"])
                + f"<p class='muted'>{summary['n']} visits &middot; "
                f"mean {_fmt(summary['mean'])} &middot; "
                f"range {_fmt(summary['range'])} &middot; "
                f"variability (CoV) {_fmt(entry['cov'])}</p>"
            )

    parts.append(
        "<h2>For the clinician</h2>"
        "<p>These are webcam-derived geometric estimates scaled by an assumed "
        "11.7&nbsp;mm iris diameter. Lid margin distances are measured to the "
        "fitted iris centre, not to a corneal light reflex, so they are a proxy "
        "for MRD1 rather than MRD1 itself. Ocular excursions are normalised "
        "aperture-relative iris displacements, not degrees of ductions. No "
        "normative data exist for any of these values. They are offered only as "
        "a structured record of what the person did at home, to sit alongside "
        "history and examination.</p>"
    )
    parts.append(
        "<footer><pre style='white-space:pre-wrap;font-family:inherit'>"
        + html.escape(disclaimers.PRIVACY_NOTE)
        + "</pre><p>"
        + html.escape(disclaimers.PERSISTENT_FOOTER)
        + f"</p><p class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M}</p></footer>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def write_report(session: dict, directory: str | None = None,
                 history: list[dict] | None = None) -> str:
    directory = storage.ensure_dir(directory)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe = "".join(
        c for c in session.get("person_label", "session") if c.isalnum() or c in "-_"
    ) or "session"
    path = os.path.join(directory, f"report_{safe}_{stamp}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_report(session, history))
    return path
