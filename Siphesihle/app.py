"""WildlifeID - Part 3 (Siphesihle): upload + results web app.

    python app.py            ->  http://127.0.0.1:5000

Flow:  POST /api/upload   validate -> resize/normalise -> store          (no model needed)
       GET  /api/results  embed with MegaDescriptor -> compare with gallery -> known / new individual

Reads the team's artifacts from the project folder (Part1_Mbali, Part2_Keightly - or the flat repo layout).
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, current_app, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

HERE = Path(__file__).resolve().parent
log = logging.getLogger("wildlifeid")

# =============================================================== model config
ARCH = "swin_large_patch4_window12_384"                 # timm name of MegaDescriptor-L-384 (finetuned profile only)
BASELINE_MODEL = os.environ.get("WILDLIFEID_BASELINE_MODEL", "BVRA/MegaDescriptor-L-384")
HF_NAME = f"hf-hub:{BASELINE_MODEL}"                     # e.g. BVRA/MegaDescriptor-T-224 on memory-constrained hosts
MODEL_INPUT_SIZES = {"BVRA/MegaDescriptor-L-384": 384, "BVRA/MegaDescriptor-T-224": 224}
ONNX_REPO = os.environ.get("WILDLIFEID_ONNX_REPO")       # set -> baseline runs via onnxruntime, not torch+timm


# =============================================================== settings
@dataclass(frozen=True)
class Settings:
    profile: str = "baseline"                 # "baseline" (Part 1) | "finetuned" (Part 2)
    project_dir: Path = HERE.parent           # folder that contains the other parts' files
    upload_dir: Path = HERE / "uploads"
    max_upload_mb: int = 10
    min_side_px: int = 32                     # some gallery crops are only 74 px
    max_pixels: int = 50_000_000              # decompression-bomb guard
    upload_ttl_hours: float = 24.0
    top_k: int = 3
    eager_load: bool = True
    similarity_threshold: float | None = None   # None -> thresholds.json
    input_size: int = 384

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        profile = e("WILDLIFEID_PROFILE", "baseline").strip().lower()
        # finetuned is always the Swin-L-384 architecture (model_weights.pt). baseline's input size
        # follows whichever model is actually serving it: the exported ONNX model when ONNX_REPO is
        # set (WILDLIFEID_ONNX_INPUT_SIZE, default 224 - our only exported model so far), otherwise
        # whichever torch model WILDLIFEID_BASELINE_MODEL points at (e.g. 384 for the full L-384).
        if profile != "baseline":
            input_size = 384
        elif ONNX_REPO:
            input_size = int(e("WILDLIFEID_ONNX_INPUT_SIZE", "224"))
        else:
            input_size = MODEL_INPUT_SIZES.get(BASELINE_MODEL, 384)
        return cls(
            profile=profile,
            project_dir=Path(e("WILDLIFEID_PROJECT", HERE.parent)),
            upload_dir=Path(e("WILDLIFEID_UPLOAD_DIR", HERE / "uploads")),
            max_upload_mb=int(e("WILDLIFEID_MAX_UPLOAD_MB", "10")),
            upload_ttl_hours=float(e("WILDLIFEID_UPLOAD_TTL_HOURS", "24")),
            top_k=int(e("WILDLIFEID_TOP_K", "3")),
            eager_load=e("WILDLIFEID_EAGER_LOAD", "1").lower() in {"1", "true", "yes"},
            similarity_threshold=float(e("WILDLIFEID_SIM_THRESHOLD")) if e("WILDLIFEID_SIM_THRESHOLD") else None,
            input_size=input_size,
        )

    # Where each profile's files live (owned by Part 1 / Part 2). Two layouts are supported:
    #   separate folders:  <project>/Part1_Mbali/..., <project>/Part2_Keightly/...   (the zip)
    #   flat repo layout:  <project>/embeddings/..., <project>/data/..., <project>/fine_tuned_embeddings.npy
    def _find(self, *candidates: str) -> Path:
        paths = [self.project_dir / c for c in candidates]
        return next((p for p in paths if p.exists()), paths[0])

    @property
    def index_csv(self) -> Path:                       # same row order for both profiles
        return self._find("Part1_Mbali/embeddings/index.csv", "embeddings/index.csv")

    @property
    def embeddings_npy(self) -> Path:
        if self.profile == "finetuned":
            return self._find("Part2_Keightly/fine_tuned_embeddings.npy", "fine_tuned_embeddings.npy")
        return self._find("Part1_Mbali/embeddings/embeddings.npy", "embeddings/embeddings.npy")

    @property
    def sample_dir(self) -> Path:
        return self._find("Part1_Mbali/data/sample", "data/sample")

    @property
    def weights_path(self) -> Path:
        if os.environ.get("WILDLIFEID_WEIGHTS"):
            return Path(os.environ["WILDLIFEID_WEIGHTS"])
        return self._find("Part2_Keightly/model_weights.pt", "model_weights.pt")

    @property
    def additions_dir(self) -> Path:          # Part 4: individuals added via the gallery admin app
        return self.project_dir / "Josephine" / "gallery_additions"


# =============================================================== preprocessing
# Must equal the Part 1 contract: RGB -> resize 384x384 (bilinear) -> /255 -> (x-0.5)/0.5
def to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    return img.convert("RGB")


def preprocess(img: Image.Image, size: int = 384) -> np.ndarray:
    """PIL image -> float32 [3, size, size] (numpy version of torchvision Resize+ToTensor+Normalize)."""
    arr = np.asarray(to_rgb(img).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return np.ascontiguousarray(((arr - 0.5) / 0.5).transpose(2, 0, 1))


# =============================================================== upload validation
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}


class UploadError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_upload"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


def validate_image(data: bytes, filename: str | None, *, min_side: int, max_pixels: int) -> Image.Image:
    """Check the file's *content* (never just its name). Returns a decoded image or raises UploadError."""
    if not data:
        raise UploadError("The file is empty.", code="empty_file")
    name = filename or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise UploadError("Unsupported file type. Please upload a JPG, PNG or WebP image.", 415, "unsupported_type")
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()                                    # full decode catches truncated files
    except Image.DecompressionBombError:
        raise UploadError("That image is too large to process.", 413, "too_many_pixels")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadError("The file is not a valid image (it may be corrupt or truncated).", 400, "corrupt_image")
    if fmt not in ALLOWED_FORMATS:
        raise UploadError("Unsupported image format. Please upload a JPG, PNG or WebP image.", 415, "unsupported_type")
    if min(img.size) < min_side:
        raise UploadError(f"The image is too small ({img.size[0]}x{img.size[1]}px). "
                          f"Each side must be at least {min_side}px.", 400, "too_small")
    return img


# =============================================================== gallery + decision
@dataclass
class Match:
    rank: int
    identity: str
    species: str
    similarity: float
    image: str            # best-matching gallery image, relative to the sample dir
    n_images: int


class Gallery:
    """Known individuals. embeddings [N,D] (unit length) <-> index.csv rows. Part 4 can extend this class."""

    def __init__(self, embeddings: np.ndarray, index: pd.DataFrame):
        if embeddings.ndim != 2 or len(embeddings) != len(index):
            raise ValueError(f"embeddings {embeddings.shape} and index ({len(index)} rows) do not line up")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = (embeddings / np.clip(norms, 1e-12, None)).astype(np.float32)
        self.index = index.reset_index(drop=True)
        self.dim = int(embeddings.shape[1])

    @classmethod
    def load(cls, embeddings_npy: Path, index_csv: Path) -> "Gallery":
        return cls(np.load(embeddings_npy), pd.read_csv(index_csv))

    def with_additions(self, additions_dir: Path, profile: str) -> "Gallery":
        """Part 4: fold in individuals added via the gallery admin app, if any exist."""
        index_csv, embeddings_npy = additions_dir / "index.csv", additions_dir / profile / "embeddings.npy"
        if not (index_csv.exists() and embeddings_npy.exists()):
            return self
        added = pd.read_csv(index_csv)
        if added.empty:
            return self
        added["file"] = "_added/" + added["file"].astype(str)         # routed by gallery_image() below
        merged_index = pd.concat([self.index, added], ignore_index=True)
        merged_embeddings = np.vstack([self.embeddings, np.load(embeddings_npy)])
        return Gallery(merged_embeddings, merged_index)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def n_identities(self) -> int:
        return int(self.index.identity.nunique())

    def search(self, query: np.ndarray, k: int = 3) -> list[Match]:
        """Top-k *individuals* by cosine similarity (best photo per individual)."""
        q = np.asarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dimension {q.shape[0]}, gallery expects {self.dim}")
        sims = self.embeddings @ (q / max(float(np.linalg.norm(q)), 1e-12))
        best: dict[str, tuple[float, int]] = {}
        for i, s in enumerate(sims):
            ident = str(self.index.identity[i])
            if ident not in best or s > best[ident][0]:
                best[ident] = (float(s), i)
        counts = self.index.identity.astype(str).value_counts()
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        return [Match(r + 1, ident, str(self.index.species[row]), sim, str(self.index.file[row]), int(counts[ident]))
                for r, (ident, (sim, row)) in enumerate(ranked)]


class LiveGallery:
    """Wraps a base Gallery and re-folds in gallery_additions/ whenever its files change, so a new
    individual added via the Part 4 admin app shows up on the next upload - no restart needed."""

    def __init__(self, base: Gallery, additions_dir: Path, profile: str):
        self._base, self._dir, self._profile = base, additions_dir, profile
        self._lock = threading.Lock()
        self._mtime = self._additions_mtime()
        self._current = base.with_additions(additions_dir, profile)

    def _additions_mtime(self) -> float | None:
        index_csv = self._dir / "index.csv"
        embeddings_npy = self._dir / self._profile / "embeddings.npy"
        if not (index_csv.exists() and embeddings_npy.exists()):
            return None
        return max(index_csv.stat().st_mtime, embeddings_npy.stat().st_mtime)

    def _refreshed(self) -> Gallery:
        mtime = self._additions_mtime()
        if mtime != self._mtime:
            with self._lock:
                if mtime != self._mtime:               # re-check: another thread may have just refreshed
                    self._current = self._base.with_additions(self._dir, self._profile)
                    self._mtime = mtime
        return self._current

    def __len__(self) -> int:
        return len(self._refreshed())

    @property
    def n_identities(self) -> int:
        return self._refreshed().n_identities

    @property
    def dim(self) -> int:
        return self._refreshed().dim

    def search(self, query: np.ndarray, k: int = 3) -> list[Match]:
        return self._refreshed().search(query, k)


def decide(matches: list[Match], threshold: float) -> dict:
    """Open-set rule: below the threshold the system reports a new individual instead of forcing a match."""
    if not matches:
        return {"decision": "new_individual", "margin": None}
    margin = matches[0].similarity - matches[1].similarity if len(matches) > 1 else None
    return {"decision": "match" if matches[0].similarity >= threshold else "new_individual",
            "margin": None if margin is None else round(margin, 4)}


# =============================================================== model


class EmbedderError(RuntimeError):
    pass


class TorchEmbedder:
    """baseline = pretrained weights from Hugging Face; finetuned = Keightly's model_weights.pt (works offline)."""
    dim = 1536            # fallback only; real value is read from the model once loaded (varies by BASELINE_MODEL)

    def __init__(self, profile: str, weights_path: Path):
        self.profile, self.weights_path = profile, Path(weights_path)
        self.name = (f"{BASELINE_MODEL} (pretrained)" if profile == "baseline" else "MegaDescriptor-L-384 (fine-tuned)")
        self._model = self._torch = self._device = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import timm
                import torch
            except ImportError as e:
                raise EmbedderError("PyTorch and timm are required: pip install torch timm") from e
            if self.profile == "finetuned":
                if not self.weights_path.exists():
                    raise EmbedderError(f"Fine-tuned weights not found at {self.weights_path}. "
                                        "Put model_weights.pt in Part2_Keightly/ (or set WILDLIFEID_WEIGHTS).")
                model = timm.create_model(ARCH, pretrained=False, num_classes=0)
                missing, unexpected = model.load_state_dict(
                    torch.load(self.weights_path, map_location="cpu", weights_only=True), strict=False)
                benign = ("relative_position_index", "attn_mask")      # non-persistent buffers in timm
                missing = [k for k in missing if not k.endswith(benign)]
                unexpected = [k for k in unexpected if not k.endswith(benign)]
                if missing or unexpected:
                    raise EmbedderError(f"Weights do not match the architecture "
                                        f"(missing={missing[:3]}, unexpected={unexpected[:3]}).")
            else:
                model = timm.create_model(HF_NAME, num_classes=0, pretrained=True)
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.eval().to(self._device)
            self.dim = int(model.num_features)      # e.g. 768 for T-224 vs 1536 for L-384

    def embed(self, x: np.ndarray) -> np.ndarray:
        """float32 [3,H,W] from preprocess() -> float32 [1536], L2-normalised."""
        if self._model is None:
            self.load()
        torch = self._torch
        with self._lock, torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(self._device)
            f = torch.nn.functional.normalize(self._model(t).float(), dim=1)
        return f[0].cpu().numpy().astype(np.float32)


class OnnxEmbedder:
    """baseline-only, onnxruntime alternative to TorchEmbedder. Importing torch+timm costs ~300MB
    of RAM before a model is even loaded - too much for memory-constrained hosts (e.g. free-tier
    PaaS). Runs a pre-exported copy of the same baseline model instead; same preprocessing in,
    same L2-normalised vector out, so Gallery/decide() don't need to know which one produced it.
    The .onnx file is pulled from WILDLIFEID_ONNX_REPO via huggingface_hub, same download-on-first-
    use pattern as the pretrained torch weights."""
    dim = 768

    def __init__(self, repo_id: str, filename: str = "model.onnx"):
        self.repo_id, self.filename = repo_id, filename
        self.name = f"{repo_id} (onnx)"
        self._session = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise EmbedderError("onnxruntime and huggingface_hub are required: "
                                    "pip install onnxruntime huggingface_hub") from e
            try:
                path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            except Exception as e:
                raise EmbedderError(f"Could not download {self.filename} from {self.repo_id}: {e}") from e
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._input_name = session.get_inputs()[0].name
            self.dim = int(session.get_outputs()[0].shape[-1])
            self._session = session

    def embed(self, x: np.ndarray) -> np.ndarray:
        """float32 [3,H,W] from preprocess() -> float32 [dim], L2-normalised."""
        if self._session is None:
            self.load()
        with self._lock:
            batch = np.ascontiguousarray(x, dtype=np.float32)[None]
            out = self._session.run(None, {self._input_name: batch})[0][0]
        norm = np.linalg.norm(out)
        return (out / norm if norm > 1e-12 else out).astype(np.float32)


def build_embedder(profile: str, weights_path: Path):
    """Single source of truth for which embedder backend serves a profile - used by this app's
    own create_app() and by Part 4's admin app, so both always agree on baseline vs onnx."""
    if profile == "baseline" and ONNX_REPO:
        return OnnxEmbedder(ONNX_REPO)
    return TorchEmbedder(profile, weights_path)


# =============================================================== upload store
_ID = re.compile(r"^[0-9a-f]{32}$")


def is_valid_id(upload_id: str) -> bool:
    return bool(_ID.match(upload_id or ""))


class UploadStore:
    """uploads/<id>/  preview.jpg (display copy, metadata stripped) | input.npy (exact model input) |
    meta.json | results.json (cached result). Folders older than the TTL are deleted."""

    def __init__(self, base: Path, ttl_hours: float):
        self.base, self.ttl = Path(base), ttl_hours * 3600
        self.base.mkdir(parents=True, exist_ok=True)

    def path(self, upload_id: str) -> Path:
        if not is_valid_id(upload_id):
            raise KeyError(upload_id)
        return self.base / upload_id

    def exists(self, upload_id: str) -> bool:
        return is_valid_id(upload_id) and (self.base / upload_id / "input.npy").exists()

    def create(self, img: Image.Image, tensor: np.ndarray, filename: str) -> str:
        self.cleanup()
        upload_id = uuid.uuid4().hex
        d = self.base / upload_id
        d.mkdir(parents=True)
        preview = to_rgb(img)
        preview.thumbnail((1024, 1024))
        preview.save(d / "preview.jpg", "JPEG", quality=90)          # re-encoding drops EXIF / GPS
        np.save(d / "input.npy", tensor)
        (d / "meta.json").write_text(json.dumps({"filename": Path(filename or "upload").name[:120],
                                                  "width": img.size[0], "height": img.size[1]}))
        return upload_id

    def meta(self, upload_id: str) -> dict:
        return json.loads((self.path(upload_id) / "meta.json").read_text())

    def load_tensor(self, upload_id: str) -> np.ndarray:
        return np.load(self.path(upload_id) / "input.npy")

    def read_results(self, upload_id: str) -> dict | None:
        p = self.path(upload_id) / "results.json"
        return json.loads(p.read_text()) if p.exists() else None

    def write_results(self, upload_id: str, results: dict) -> None:
        (self.path(upload_id) / "results.json").write_text(json.dumps(results))

    def cleanup(self) -> int:
        removed, now = 0, time.time()
        for d in self.base.iterdir():
            if d.is_dir() and is_valid_id(d.name) and now - d.stat().st_mtime > self.ttl:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed


# =============================================================== app
@dataclass
class Services:
    settings: Settings
    store: UploadStore
    gallery: Gallery | LiveGallery
    embedder: object
    threshold: float
    threshold_source: str


def _load_threshold(settings: Settings) -> tuple[float, str]:
    data = json.loads((HERE / "thresholds.json").read_text())[settings.profile]
    return float(data["similarity_threshold"]), data["source"]


def create_app(settings: Settings | None = None, *, embedder=None, gallery: Gallery | None = None,
               threshold: float | None = None) -> Flask:
    """App factory. Tests inject a fake embedder; production builds the real one."""
    settings = settings or Settings.from_env()
    if settings.profile not in {"baseline", "finetuned"}:
        raise ValueError("WILDLIFEID_PROFILE must be 'baseline' or 'finetuned'")
    gallery = gallery or LiveGallery(Gallery.load(settings.embeddings_npy, settings.index_csv),
                                     settings.additions_dir, settings.profile)

    if threshold is not None:
        thr, src = threshold, "argument"
    elif settings.similarity_threshold is not None:
        thr, src = settings.similarity_threshold, "WILDLIFEID_SIM_THRESHOLD"
    else:
        thr, src = _load_threshold(settings)

    if embedder is None:
        embedder = build_embedder(settings.profile, settings.weights_path)
        if settings.eager_load:
            try:
                embedder.load()
            except Exception as e:      # uploads still work; /api/results answers 503 with the reason
                log.error("Model failed to load: %s", e)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = (settings.max_upload_mb + 1) * 1024 * 1024
    app.extensions["wildlifeid"] = Services(settings, UploadStore(settings.upload_dir, settings.upload_ttl_hours),
                                            gallery, embedder, thr, src)

    def svc() -> Services:
        return current_app.extensions["wildlifeid"]

    def err(code: str, message: str, status: int):
        return jsonify({"error": {"code": code, "message": message}}), status

    # ---- pages
    @app.get("/")
    def index():
        return render_template("index.html", max_mb=settings.max_upload_mb, profile=settings.profile)

    @app.get("/results/<upload_id>")
    def results_page(upload_id):
        if not svc().store.exists(upload_id):
            abort(404, "That upload was not found. It may have expired - please upload the photo again.")
        return render_template("results.html", upload_id=upload_id)

    # ---- API
    @app.post("/api/upload")
    def api_upload():
        f = request.files.get("file")
        if f is None:
            raise UploadError("No file was sent. Use multipart/form-data with a field named 'file'.", 400, "no_file")
        img = validate_image(f.read(), f.filename, min_side=settings.min_side_px, max_pixels=settings.max_pixels)
        upload_id = svc().store.create(img, preprocess(img, settings.input_size), f.filename or "upload")
        return jsonify({
            "upload_id": upload_id, "filename": svc().store.meta(upload_id)["filename"],
            "width": img.size[0], "height": img.size[1],
            "preview_url": url_for("upload_preview", upload_id=upload_id),
            "results_url": url_for("api_results", upload_id=upload_id),
            "results_page_url": url_for("results_page", upload_id=upload_id)}), 201

    @app.get("/api/results/<upload_id>")
    def api_results(upload_id):
        s = svc()
        if not s.store.exists(upload_id):
            return err("not_found", "Upload not found or expired.", 404)
        if (cached := s.store.read_results(upload_id)):
            return jsonify(cached)
        try:
            t0 = time.perf_counter()
            emb = s.embedder.embed(s.store.load_tensor(upload_id))
            embed_ms = (time.perf_counter() - t0) * 1000
        except EmbedderError as e:
            return err("model_unavailable", str(e), 503)
        except Exception as e:
            current_app.logger.exception("inference failed")
            return err("inference_failed", "The model could not run: " + str(e), 503)

        matches = s.gallery.search(emb, settings.top_k)
        verdict = decide(matches, s.threshold)
        results = {
            "upload_id": upload_id, "filename": s.store.meta(upload_id)["filename"],
            "preview_url": url_for("upload_preview", upload_id=upload_id),
            "decision": verdict["decision"], "margin": verdict["margin"],
            "threshold": round(s.threshold, 4), "threshold_source": s.threshold_source,
            "metric": "cosine similarity (higher = more alike)",
            "matches": [{"rank": m.rank, "identity": m.identity, "species": m.species,
                         "similarity": round(m.similarity, 4), "n_gallery_images": m.n_images,
                         "above_threshold": m.similarity >= s.threshold,
                         "image_url": url_for("gallery_image", filename=m.image)} for m in matches],
            "model": {"profile": settings.profile, "name": s.embedder.name, "embedding_dim": s.gallery.dim},
            "gallery": {"images": len(s.gallery), "individuals": s.gallery.n_identities},
            "timing_ms": {"embed": round(embed_ms, 1)},
        }
        s.store.write_results(upload_id, results)
        return jsonify(results)

    @app.get("/api/health")
    def api_health():
        s = svc()
        return jsonify({"status": "ok", "profile": settings.profile, "model": s.embedder.name,
                        "model_loaded": bool(getattr(s.embedder, "ready", True)),
                        "gallery_images": len(s.gallery), "gallery_individuals": s.gallery.n_identities,
                        "embedding_dim": s.gallery.dim, "similarity_threshold": s.threshold,
                        "max_upload_mb": settings.max_upload_mb})

    # ---- files for the UI
    @app.get("/uploads/<upload_id>/preview.jpg")
    def upload_preview(upload_id):
        if not svc().store.exists(upload_id):
            abort(404)
        return send_from_directory(svc().store.path(upload_id), "preview.jpg", mimetype="image/jpeg", max_age=3600)

    @app.get("/gallery-images/<path:filename>")
    def gallery_image(filename):
        if filename.startswith("_added/"):            # Part 4: individual added via the gallery admin app
            return send_from_directory(settings.additions_dir, filename[len("_added/"):], max_age=86400)
        return send_from_directory(settings.sample_dir, filename, max_age=86400)    # safe_join inside

    # ---- errors (JSON for /api, normal pages otherwise)
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
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
                     debug=False, threaded=True)
