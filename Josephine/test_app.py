"""Tests for Part 4.  Run:  python -m pytest -q test_app.py   (fake embedders, no model download)."""
import io

import numpy as np
import pytest
from PIL import Image

from app import AdminSettings, Gallery, create_app


class FakeEmbedder:
    dim = 1536

    def __init__(self, seed=0):
        self.seed = seed
        self.calls = 0

    def embed(self, x):
        self.calls += 1
        assert x.shape == (3, 384, 384) and x.dtype == np.float32
        return np.random.default_rng(self.seed).normal(size=1536).astype(np.float32)


@pytest.fixture
def settings(tmp_path):
    return AdminSettings(additions_dir=tmp_path / "gallery_additions", max_upload_mb=1)


@pytest.fixture
def embedders():
    return {"baseline": FakeEmbedder(1), "finetuned": FakeEmbedder(2)}


@pytest.fixture
def app(settings, embedders):
    return create_app(settings, embedders=embedders)


@pytest.fixture
def client(app):
    return app.test_client()


def img_buf(size=(200, 200), color=(10, 120, 40), fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    buf.seek(0)
    return buf


def add(client, identity="Zebra_900", species="zebra", files=None):
    files = files or [(img_buf(), "a.jpg")]
    return client.post("/api/gallery", data={"identity": identity, "species": species, "files": files},
                       content_type="multipart/form-data")


# ------------------------------------------------------------------ pages / health
def test_admin_page(client):
    r = client.get("/admin")
    assert r.status_code == 200 and b"Known individuals" in r.data


def test_health(client):
    r = client.get("/api/health").get_json()
    assert r["status"] == "ok" and r["base_individuals"] == 8 and r["added_individuals"] == 0


# ------------------------------------------------------------------------- list
def test_list_shows_the_8_base_identities(client):
    r = client.get("/api/gallery").get_json()
    assert r["total"] == 8
    assert all(i["source"] == "baseline" and i["n_images"] == 3 for i in r["individuals"])
    assert client.get(r["individuals"][0]["sample_image_url"]).status_code == 200


# ------------------------------------------------------------------------- add
def test_add_then_appears_in_list(client):
    r = add(client, "Zebra_900", "zebra")
    assert r.status_code == 201
    d = r.get_json()
    assert d == {"identity": "Zebra_900", "species": "zebra", "n_images": 1}

    listing = client.get("/api/gallery").get_json()
    assert listing["total"] == 9
    added = next(i for i in listing["individuals"] if i["source"] == "added")
    assert added == {"identity": "Zebra_900", "species": "zebra", "n_images": 1,
                     "source": "added", "sample_image_url": added["sample_image_url"]}
    assert client.get(added["sample_image_url"]).status_code == 200


def test_add_multiple_photos_counts_them(client):
    r = add(client, "Zebra_901", "zebra", files=[(img_buf(), "a.jpg"), (img_buf(), "b.jpg")])
    assert r.status_code == 201 and r.get_json()["n_images"] == 2


def test_add_embeds_under_both_profiles(client, embedders, settings):
    add(client, "Zebra_902", "zebra")
    assert embedders["baseline"].calls == 1 and embedders["finetuned"].calls == 1
    assert (settings.additions_dir / "baseline" / "embeddings.npy").exists()
    assert (settings.additions_dir / "finetuned" / "embeddings.npy").exists()


def test_add_duplicate_identity_rejected(client):
    add(client, "Zebra_903", "zebra")
    r = add(client, "Zebra_903", "zebra")
    assert r.status_code == 409 and r.get_json()["error"]["code"] == "duplicate_identity"


def test_add_duplicate_of_base_identity_rejected(client):
    r = add(client, "DogFaceNet_208", "dog")          # already one of the 8 base identities
    assert r.status_code == 409 and r.get_json()["error"]["code"] == "duplicate_identity"


def test_add_rejections(client):
    cases = [
        ({"identity": "", "species": "zebra"}, [(img_buf(), "a.jpg")], 400, "invalid_identity"),
        ({"identity": "../etc", "species": "zebra"}, [(img_buf(), "a.jpg")], 400, "invalid_identity"),
        ({"identity": "Zebra_904", "species": ""}, [(img_buf(), "a.jpg")], 400, "invalid_species"),
        ({"identity": "Zebra_905", "species": "zebra"}, [], 400, "no_file"),
        ({"identity": "Zebra_906", "species": "zebra"}, [(io.BytesIO(b"not an image"), "a.jpg")], 400, "corrupt_image"),
    ]
    for form, files, status, code in cases:
        data = dict(form, files=files) if files else form
        r = client.post("/api/gallery", data=data, content_type="multipart/form-data")
        assert (r.status_code, r.get_json()["error"]["code"]) == (status, code), form


def test_add_too_large(client):                       # test app limit is 1 MB
    import os
    big = io.BytesIO(os.urandom(2 * 1024 * 1024))
    r = client.post("/api/gallery", data={"identity": "Zebra_907", "species": "zebra", "files": (big, "big.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 413 and r.get_json()["error"]["code"] == "file_too_large"


# -------------------------------------------------------- integration with Part 3
def test_gallery_with_additions_merges_into_part3(client, settings):
    """Proves Part 3's Gallery actually picks up what the admin app writes (the whole point
    of the shared gallery_additions/ store)."""
    add(client, "Zebra_908", "zebra")

    base = Gallery.load(settings.base.embeddings_npy, settings.base.index_csv)
    merged = base.with_additions(settings.additions_dir, "baseline")
    assert len(merged) == 25 and merged.n_identities == 9
    added_row = merged.index[merged.index.identity == "Zebra_908"].iloc[0]
    assert added_row.file.startswith("_added/images/")


def test_gallery_with_additions_is_noop_when_empty(tmp_path):
    from app import Settings
    base = Gallery.load(Settings().embeddings_npy, Settings().index_csv)
    merged = base.with_additions(tmp_path / "does_not_exist", "baseline")
    assert len(merged) == 24 and merged.n_identities == 8
