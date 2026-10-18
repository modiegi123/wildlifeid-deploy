"""WildlifeID - combined entrypoint for all four parts of the project.

Runs Part 3 (Siphesihle: upload + results), Part 4 (Josephine: gallery admin)
and the Pipeline report (Part 1 Palesa + Part 2 Keightly results) as one
process on one port, instead of three separate Flask apps on three separate
ports. Each part keeps owning its own app.py/templates/tests - this just
mounts all three WSGI apps behind one dev server using Werkzeug's
DispatcherMiddleware, so url_for()/fetch() inside each app still resolve
correctly under their mount prefix.

    python run.py                 ->  http://127.0.0.1:5000/            upload + results (Part 3, Siphesihle)
                                       http://127.0.0.1:5000/admin       gallery admin    (Part 4, Josephine)
                                       http://127.0.0.1:5000/pipeline    Part 1 (Palesa) + Part 2 (Keightly) report

Env vars (same as before): WILDLIFEID_PROFILE, WILDLIFEID_PROJECT, HOST, PORT.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

HERE = Path(__file__).resolve().parent


def _load_app_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_app():
    siphesihle = _load_app_module("wildlifeid_part3_app", HERE / "Siphesihle" / "app.py")
    josephine = _load_app_module("wildlifeid_part4_app", HERE / "Josephine" / "app.py")
    pipeline = _load_app_module("wildlifeid_pipeline_app", HERE / "Pipeline" / "app.py")

    main_app = siphesihle.create_app()
    admin_app = josephine.create_app()
    pipeline_app = pipeline.create_app()

    return DispatcherMiddleware(main_app, {"/admin": admin_app, "/pipeline": pipeline_app})


if __name__ == "__main__":
    app = build_app()
    run_simple(os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5000")),
               app, threaded=True)
