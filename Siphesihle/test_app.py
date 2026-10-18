"""Tests for Part 3.  Run:  python -m pytest -q test_app.py   (no GPU / model download needed: a fake embedder is used)."""
import io
import os
import time

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app import EmbedderError, Gallery, Match, Settings, create_app, decide

class FakeEmbedder:
    """Returns whatever vector the test sets, and counts calls."""
    name = "fake"
    dim = 1536
    ready = True

    def __init__(self):
        self.vector = None
        self.calls = 0
        self.error = None

    def embed(self, x):
        self.calls += 1
        if self.error:
            raise self.error
        assert x.shape == (3, 384, 384) and x.dtype == np.float32
        return self.vector


@pytest.fixture
def gallery():
    return Gallery.load(Settings().embeddings_npy, Settings().index_csv)


@pytest.fixture
def fake():
    return FakeEmbedder()


@pytest.fixture
def app(tmp_path, fake, gallery):
    settings = Settings(profile="baseline", upload_dir=tmp_path / "uploads", max_upload_mb=1)
    return create_app(settings, embedder=fake, gallery=gallery, threshold=0.1985)


@pytest.fixture
def client(app):
    return app.test_client()


def make_image(size=(300, 200), fmt="JPEG", mode="RGB", color=(120, 160, 90), **save_kw):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, fmt, **save_kw)
    buf.seek(0)
    return buf


def post_file(client, buf, name="photo.jpg"):
    return client.post("/api/upload", data={"file": (buf, name)}, content_type="multipart/form-data")


# ------------------------------------------------------------------ basic pages / health
def test_health(client):
    r = client.get("/api/health").get_json()
    assert r["status"] == "ok" and r["gallery_images"] == 24 and r["gallery_individuals"] == 8
    assert r["embedding_dim"] == 1536


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Who is this animal?" in r.data


# ------------------------------------------------------------------------------ upload
def test_upload_ok_stores_normalised_tensor_and_preview(client, app):
    r = post_file(client, make_image((300, 200)))
    assert r.status_code == 201
    d = r.get_json()
    assert len(d["upload_id"]) == 32 and d["width"] == 300 and d["height"] == 200
    svc = app.extensions["wildlifeid"]
    x = svc.store.load_tensor(d["upload_id"])
    assert x.shape == (3, 384, 384) and x.dtype == np.float32
    assert -1.0001 <= x.min() and x.max() <= 1.0001            # (pixel/255 - .5) / .5
    prev = client.get(d["preview_url"])
    assert prev.status_code == 200 and prev.mimetype == "image/jpeg"


def test_upload_png_with_alpha_and_webp(client):
    assert post_file(client, make_image((100, 100), "PNG", "RGBA", (10, 200, 10, 0)), "a.png").status_code == 201
    assert post_file(client, make_image((100, 120), "WEBP"), "a.webp").status_code == 201


def test_upload_strips_exif_gps(client, app):
    img = Image.new("RGB", (200, 200), (1, 2, 3))
    exif = Image.Exif()
    exif[0x8825] = {1: "S", 2: (26.0, 12.0, 0.0)}               # GPSInfo
    exif[0x010F] = "SecretCam"
    buf = io.BytesIO(); img.save(buf, "JPEG", exif=exif); buf.seek(0)
    d = post_file(client, buf).get_json()
    saved = Image.open(app.extensions["wildlifeid"].store.path(d["upload_id"]) / "preview.jpg")
    assert not saved.getexif()


def test_upload_applies_exif_rotation(client, app):
    img = Image.new("RGB", (200, 100), (200, 0, 0))
    exif = Image.Exif(); exif[0x0112] = 6                       # rotate 90 deg clockwise to display
    buf = io.BytesIO(); img.save(buf, "JPEG", exif=exif); buf.seek(0)
    d = post_file(client, buf).get_json()
    prev = Image.open(app.extensions["wildlifeid"].store.path(d["upload_id"]) / "preview.jpg")
    assert prev.size == (100, 200)


def test_upload_rejections(client):
    cases = [
        (io.BytesIO(b""), "empty.jpg", 400, "empty_file"),
        (io.BytesIO(b"hello"), "notes.txt", 415, "unsupported_type"),
        (io.BytesIO(b"this is not an image"), "fake.jpg", 400, "corrupt_image"),
        (make_image((50, 50), "GIF"), "anim.gif", 415, "unsupported_type"),
        (make_image((50, 50), "GIF"), "disguised.jpg", 415, "unsupported_type"),   # content, not name
        (make_image((20, 20)), "tiny.jpg", 400, "too_small"),
    ]
    for buf, name, status, code in cases:
        r = post_file(client, buf, name)
        assert (r.status_code, r.get_json()["error"]["code"]) == (status, code), name


def test_upload_truncated_jpeg(client):
    data = make_image((600, 600)).getvalue()
    r = post_file(client, io.BytesIO(data[: len(data) // 2]))
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "corrupt_image"


def test_upload_no_file_field(client):
    r = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "no_file"


def test_upload_too_large(client):                              # test app limit is 1 MB
    r = post_file(client, io.BytesIO(os.urandom(3 * 1024 * 1024)), "big.jpg")
    assert r.status_code == 413 and r.get_json()["error"]["code"] == "file_too_large"


# ----------------------------------------------------------------------------- results
def _upload(client):
    return post_file(client, make_image()).get_json()["upload_id"]


def test_results_known_individual(client, fake, gallery):
    fake.vector = gallery.embeddings[3] * 0.98 + gallery.embeddings[4] * 0.02    # a DogFaceNet photo
    uid = _upload(client)
    r = client.get(f"/api/results/{uid}")
    d = r.get_json()
    assert r.status_code == 200 and d["decision"] == "match"
    assert d["matches"][0]["identity"] == "DogFaceNet_208" and d["matches"][0]["similarity"] > 0.99
    assert [m["rank"] for m in d["matches"]] == [1, 2, 3]
    assert len({m["identity"] for m in d["matches"]}) == 3       # aggregated per individual
    assert d["matches"][0]["n_gallery_images"] == 3 and d["matches"][0]["above_threshold"] is True
    assert client.get(d["matches"][0]["image_url"]).status_code == 200
    assert d["threshold"] == 0.1985 and d["gallery"] == {"images": 24, "individuals": 8}


def test_results_new_individual(client, fake, gallery):
    rng = np.random.default_rng(0)
    v = rng.normal(size=1536).astype(np.float32)
    G = gallery.embeddings
    v -= G.T @ np.linalg.lstsq(G.T, v, rcond=None)[0]            # remove everything the gallery can explain
    fake.vector = v / np.linalg.norm(v)
    d = client.get(f"/api/results/{_upload(client)}").get_json()
    assert d["decision"] == "new_individual" and d["matches"][0]["above_threshold"] is False


def test_results_are_cached(client, fake, gallery):
    fake.vector = gallery.embeddings[0]
    uid = _upload(client)
    a = client.get(f"/api/results/{uid}").get_json()
    b = client.get(f"/api/results/{uid}").get_json()
    assert a == b and fake.calls == 1


def test_results_unknown_and_malicious_ids(client):
    assert client.get("/api/results/" + "0" * 32).status_code == 404
    for bad in ["..%2F..%2Fetc%2Fpasswd", "not-an-id", "A" * 32]:
        assert client.get(f"/api/results/{bad}").status_code == 404
    assert client.get("/results/" + "0" * 32).status_code == 404
    assert client.get("/uploads/" + "0" * 32 + "/preview.jpg").status_code == 404


def test_results_page_for_real_upload(client):
    uid = _upload(client)
    r = client.get(f"/results/{uid}")
    assert r.status_code == 200 and uid.encode() in r.data


def test_model_unavailable_returns_503(client, fake):
    fake.error = EmbedderError("weights missing")
    r = client.get(f"/api/results/{_upload(client)}")
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "model_unavailable"


def test_gallery_image_path_traversal_blocked(client):
    assert client.get("/gallery-images/images/000_ZindiTurtleRecall_t_id_15bo4NKD.jpg").status_code == 200
    for p in ["../baseline/embeddings.npy", "%2e%2e/baseline/embeddings.npy", "../../run.py"]:
        assert client.get("/gallery-images/" + p).status_code == 404


def test_cleanup_removes_old_uploads(app, client):
    uid = _upload(client)
    svc = app.extensions["wildlifeid"]
    old = time.time() - 3 * 24 * 3600
    os.utime(svc.store.path(uid), (old, old))
    assert svc.store.cleanup() == 1 and not svc.store.exists(uid)


def test_decide_threshold_boundary():
    ms = [Match(1, "a", "x", 0.30, "i", 3), Match(2, "b", "x", 0.10, "i", 3)]
    assert decide(ms, 0.30)["decision"] == "match"                # inclusive
    assert decide(ms, 0.31)["decision"] == "new_individual"
    assert decide(ms, 0.30)["margin"] == 0.2
    assert decide([], 0.3)["decision"] == "new_individual"


def test_search_aggregates_per_identity_and_normalises():
    E = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32)
    idx = pd.DataFrame({"row": [0, 1, 2], "file": ["a", "b", "c"], "identity": ["A", "A", "B"],
                        "species": ["s", "s", "t"]})
    g = Gallery(E, idx)
    res = g.search(np.array([5.0, 0, 0]), k=5)                   # un-normalised query is fine
    assert [m.identity for m in res] == ["A", "B"] and res[0].image == "a" and res[0].n_images == 2
    assert abs(res[0].similarity - 1.0) < 1e-6
