"""WildlifeID - Pipeline report: Part 1 (Palesa) + Part 2 (Keightly) results.

Parts 1 and 2 are offline steps (a script and a Colab notebook), not web apps -
there's nothing to "upload a photo" to. This page is the one that actually
reads and shows their output so the whole four-part pipeline (Palesa's baseline
embeddings -> Keightly's fine-tuning -> Siphesihle's upload/match app ->
Josephine's gallery admin) is visible from one project, not just two of its
four parts.

    python app.py            ->  http://127.0.0.1:5002/

It reads (does not own):
Part 1 (Palesa):   ../embeddings/{model_card.json,baseline_eval.json,similarity.png}
Part 2 (Keightly): ../config.json, ../fine_tuned_eval.json, ../comparison_metrics.csv,
                    ../oscr_curve.png
Part 3+4 (Siphesihle + Josephine): ../Siphesihle/thresholds.json,
                    ../Josephine/evaluation_results.json (the real, deployed-pipeline
                    comparison of both profiles - see ../Josephine/EVALUATION.md)
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _read_comparison_csv(path: Path):
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def build_report() -> dict:
    model_card = _read_json(ROOT / "embeddings" / "model_card.json") or {}
    baseline_eval = _read_json(ROOT / "embeddings" / "baseline_eval.json") or {}
    finetune_config = _read_json(ROOT / "config.json") or {}
    finetune_eval = _read_json(ROOT / "fine_tuned_eval.json") or {}
    comparison_rows = _read_comparison_csv(ROOT / "comparison_metrics.csv")
    thresholds = _read_json(ROOT / "Siphesihle" / "thresholds.json") or {}
    deployed_eval = _read_json(ROOT / "Josephine" / "evaluation_results.json") or {}

    return {
        "part1": {"model_card": model_card, "eval": baseline_eval,
                  "has_image": (ROOT / "embeddings" / "similarity.png").exists()},
        "part2": {"config": finetune_config, "eval": finetune_eval,
                  "has_image": (ROOT / "oscr_curve.png").exists()},
        "comparison_rows": comparison_rows,
        "thresholds": thresholds,
        "deployed_summary": deployed_eval.get("summary", deployed_eval if isinstance(deployed_eval, dict) else {}),
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def report():
        return render_template("report.html", **build_report())

    @app.get("/api/report")
    def api_report():
        return jsonify(build_report())

    @app.get("/images/<path:filename>")
    def pipeline_image(filename):
        for base in (ROOT / "embeddings", ROOT):
            if (base / filename).exists():
                return send_from_directory(base, filename, max_age=86400)
        abort(404)

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5002")),
                      debug=False, threaded=True)
