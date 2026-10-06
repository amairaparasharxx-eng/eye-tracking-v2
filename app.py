from flask import Flask, jsonify, render_template, request
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np

from metrics import measure_frame
from landmarks import FaceFrame

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/contact")
def api_contact():
    data = request.get_json(silent=True) or {}
    contact = str(data.get("contact") or "").strip()
    if not contact:
        return jsonify({"ok": True, "saved": False})
    if len(contact) > 254:
        return jsonify({"error": "Contact information is too long."}), 400

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contact": contact,
    }
    try:
        with Path("contact_submissions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return jsonify({"ok": True, "saved": True})
    except OSError:
        return jsonify({"error": "Could not save contact information."}), 500


@app.post("/api/measure")
def api_measure():
    data = request.get_json(silent=True) or {}
    points = data.get("points")
    width = int(data.get("width") or 0)
    height = int(data.get("height") or 0)
    timestamp = float(data.get("timestamp") or 0.0)

    if not points or width <= 0 or height <= 0:
        return jsonify({"error": "Invalid landmark data."}), 400

    try:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 478 or pts.shape[1] != 2:
            raise ValueError("Expected at least 478 two-dimensional landmarks.")

        face = FaceFrame(
            ok=True,
            points=pts,
            width=width,
            height=height,
            timestamp=timestamp,
        )
        result = measure_frame(face)
        return jsonify(result.as_dict())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
