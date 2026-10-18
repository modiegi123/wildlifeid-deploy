"""WildlifeID - Part 4 (Josephine): gallery admin (add/list known individuals).

    python app.py            ->  http://127.0.0.1:5001/admin

New individuals are embedded under BOTH profiles (baseline + fine-tuned) and written to
gallery_additions/, a shared on-disk store that Siphesihle's app.py (Part 3) also reads -
so anything added here shows up as a possible match the next time someone uploads a photo
there, regardless of which profile her app is running.

Reuses Settings / Gallery / TorchEmbedder / preprocess / validate_image / UploadError from
Siphesihle's app.py (sibling folder) so preprocessing and validation stay identical across
parts - see ../Siphesihle/app.py.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, current_app, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image
from werkzeug.exceptions import HTTPException

HERE = Path(__file__).resolve().parent
SIPHESIHLE_DIR = HERE.parent / "Siphesihle"


def _load_siphesihle_app():
    """Both files are named app.py, so a plain `from app import ...` after a sys.path
    insert collides with this module's own name. Load it under a distinct module name
    instead (keeps __file__ correct inside it, so its own HERE/Path logic still works)."""
    name = "wildlifeid_part3_app"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SIPHESIHLE_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_part3 = _load_siphesihle_app()
EmbedderError = _part3.EmbedderError
Gallery = _part3.Gallery
Settings = _part3.Settings
TorchEmbedder = _part3.TorchEmbedder
UploadError = _part3.UploadError
decide = _part3.decide
preprocess = _part3.preprocess
validate_image = _part3.validate_image

log = logging.getLogger("wildlifeid.admin")

PROFILES = ("baseline", "finetuned")
ADDITIONS_COLUMNS = ["row", "file", "identity", "species"]
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-]{0,63}$")


# =============================================================== settings
@dataclass(frozen=True)
class AdminSettings:
    base: Settings = field(default_factory=Settings.from_env)
    additions_dir: Path = HERE / "gallery_additions"     # must match Settings.additions_dir in Part 3's app.py
    max_upload_mb: int = 10
    eager_load: bool = False          # embedders load lazily on first /api/gallery POST

    @classmethod
    def from_env(cls) -> "AdminSettings":
        e = os.environ.get
        return cls(additions_dir=Path(e("WILDLIFEID_ADDITIONS_DIR", HERE / "gallery_additions")),
                   max_upload_mb=int(e("WILDLIFEID_MAX_UPLOAD_MB", "10")),
                   eager_load=e("WILDLIFEID_ADMIN_EAGER_LOAD", "0").lower() in {"1", "true", "yes"})


# =============================================================== additions store
def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(columns=ADDITIONS_COLUMNS)


def load_additions_index(additions_dir: Path) -> pd.DataFrame:
    index_csv = additions_dir / "index.csv"
    if index_csv.exists():
        return pd.read_csv(index_csv)
    return _empty_index()


def load_additions_embeddings(additions_dir: Path, profile: str, dim: int) -> np.ndarray:
    p = additions_dir / profile / "embeddings.npy"
    if p.exists():
        return np.load(p)
    return np.zeros((0, dim), dtype=np.float32)


def known_identities(base_index: pd.DataFrame, additions_dir: Path) -> set[str]:
    return set(base_index.identity.astype(str)) | set(load_additions_index(additions_dir).identity.astype(str))


class AdditionsStore:
    """Appends a new individual's photos + per-profile embeddings. One writer at a time."""

    def __init__(self, additions_dir: Path):
        self.dir = additions_dir
        self.images_dir = additions_dir / "images"
        self._lock = threading.Lock()
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def append(self, identity: str, species: str, images: list[Image.Image],
               tensors: list[np.ndarray], embedders: dict[str, "TorchEmbedder"]) -> int:
        with self._lock:
            index = load_additions_index(self.dir)
            saved_files = []
            for img in images:
                filename = f"{uuid.uuid4().hex}.jpg"
                img.convert("RGB").save(self.images_dir / filename, "JPEG", quality=90)
                saved_files.append(f"images/{filename}")
            new_rows = pd.DataFrame({"row": 0, "file": saved_files,
                                     "identity": identity, "species": species})
            index = pd.concat([index, new_rows], ignore_index=True)
            index["row"] = range(len(index))
            index.to_csv(self.dir / "index.csv", index=False)

            for profile, embedder in embedders.items():
                vecs = np.stack([embedder.embed(t) for t in tensors]).astype(np.float32)
                existing = load_additions_embeddings(self.dir, profile, embedder.dim)
                merged = np.vstack([existing, vecs])
                out_dir = self.dir / profile
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "embeddings.npy", merged)
            return len(saved_files)


# =============================================================== app
@dataclass
class Services:
    settings: AdminSettings
    store: AdditionsStore
    embedders: dict             # profile -> TorchEmbedder | None, populated lazily
    embedders_lock: threading.Lock


def _get_embedder(svc: Services, profile: str) -> TorchEmbedder:
    if svc.embedders.get(profile) is None:
        with svc.embedders_lock:
            if svc.embedders.get(profile) is None:
                weights = Settings(profile=profile).weights_path
                embedder = _part3.build_embedder(profile, weights)
                embedder.load()
                svc.embedders[profile] = embedder
    return svc.embedders[profile]


def create_app(settings: AdminSettings | None = None, *, embedders: dict | None = None) -> Flask:
    """App factory. Tests inject fake embedders keyed by profile name."""
    settings = settings or AdminSettings.from_env()
    store = AdditionsStore(settings.additions_dir)
    services = Services(settings, store, dict(embedders) if embedders else {}, threading.Lock())

    if settings.eager_load and embedders is None:
        for profile in PROFILES:
            try:
                _get_embedder(services, profile)
            except Exception as e:
                log.error("Could not pre-load %s embedder: %s", profile, e)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = (settings.max_upload_mb + 1) * 1024 * 1024
    app.extensions["wildlifeid_admin"] = services

    def svc() -> Services:
        return current_app.extensions["wildlifeid_admin"]

    def err(code: str, message: str, status: int):
        return jsonify({"error": {"code": code, "message": message}}), status

    # ---- pages
    @app.get("/")
    def admin():
        base_index = pd.read_csv(settings.base.index_csv)
        return render_template("admin.html", base_count=int(base_index.identity.nunique()))

    # ---- API
    @app.get("/api/gallery")
    def api_gallery_list():
        base_index = pd.read_csv(settings.base.index_csv)
        rows = []
        for identity, grp in base_index.groupby("identity", sort=False):
            first = grp.iloc[0]
            rows.append({"identity": str(identity), "species": str(first.species),
                        "n_images": int(len(grp)), "source": "baseline",
                        "sample_image_url": url_for("base_gallery_image", filename=first.file)})
        add_index = load_additions_index(settings.additions_dir)
        for identity, grp in add_index.groupby("identity", sort=False):
            first = grp.iloc[0]
            rows.append({"identity": str(identity), "species": str(first.species),
                        "n_images": int(len(grp)), "source": "added",
                        "sample_image_url": url_for("gallery_image", filename=first.file)})
        return jsonify({"individuals": rows, "total": len(rows)})

    @app.post("/api/gallery")
    def api_gallery_add():
        s = svc()
        identity = (request.form.get("identity") or "").strip()
        species = (request.form.get("species") or "").strip()
        files = [f for f in request.files.getlist("files") if f and f.filename]

        if not identity or not _IDENTITY_RE.match(identity):
            raise UploadError("Identity must be 1-64 letters, numbers, spaces, - or _, "
                              "and start with a letter or number.", 400, "invalid_identity")
        if not species:
            raise UploadError("Species is required.", 400, "invalid_species")
        if not files:
            raise UploadError("At least one photo is required.", 400, "no_file")
        base_index = pd.read_csv(settings.base.index_csv)
        if identity.lower() in {i.lower() for i in known_identities(base_index, settings.additions_dir)}:
            raise UploadError(f"'{identity}' is already in the gallery.", 409, "duplicate_identity")

        images, tensors = [], []
        for f in files:
            img = validate_image(f.read(), f.filename,
                                 min_side=settings.base.min_side_px, max_pixels=settings.base.max_pixels)
            images.append(img)
            tensors.append(preprocess(img, settings.base.input_size))

        embedders, unavailable = {}, {}
        for p in PROFILES:
            try:
                embedders[p] = _get_embedder(s, p)
            except EmbedderError as e:
                unavailable[p] = str(e)
        if not embedders:
            return err("model_unavailable", "; ".join(unavailable.values()), 503)

        n = s.store.append(identity, species, images, tensors, embedders)
        result = {"identity": identity, "species": species, "n_images": n}
        if unavailable:        # only mentioned when something actually degraded
            result["skipped_profiles"] = unavailable
        return jsonify(result), 201

    @app.get("/api/health")
    def api_health():
        base_index = pd.read_csv(settings.base.index_csv)
        add_index = load_additions_index(settings.additions_dir)
        loaded = {p: svc().embedders.get(p) is not None for p in PROFILES}
        return jsonify({"status": "ok",
                        "base_individuals": int(base_index.identity.nunique()),
                        "added_individuals": int(add_index.identity.nunique()) if len(add_index) else 0,
                        "embedders_loaded": loaded})

    # ---- files for the UI
    @app.get("/base-gallery-images/<path:filename>")
    def base_gallery_image(filename):
        return send_from_directory(settings.base.sample_dir, filename, max_age=86400)

    @app.get("/gallery-images/<path:filename>")
    def gallery_image(filename):
        return send_from_directory(settings.additions_dir, filename, max_age=86400)   # filename already "images/..."

    # ---- errors
    @app.errorhandler(UploadError)
    def _upload_error(e):
        return err(e.code, e.message, e.status)

    @app.errorhandler(413)
    def _too_large(_e):
        return err("file_too_large", f"The file is larger than the {settings.max_upload_mb} MB limit.", 413)

    @app.errorhandler(HTTPException)
    def _http_error(e):
        if request.path.startswith("/api/"):
            return err(e.name.lower().replace(" ", "_"), e.description or e.name, e.code or 500)
        return e

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5001")),
                     debug=False, threaded=True)
