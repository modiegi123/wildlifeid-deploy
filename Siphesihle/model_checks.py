"""Two helper checks for the team.

  python model_checks.py calibrate      # recompute thresholds.json from the labelled sample embeddings
  python model_checks.py smoke baseline # run the REAL model on the 24 sample photos (needs torch + timm + internet once)
  python model_checks.py smoke finetuned# same, with Keightly's model_weights.pt (works offline)

calibrate: leave-one-identity-out. "known" = best similarity to another photo of the SAME animal;
"unknown" = best similarity to any OTHER animal (its own identity removed from the gallery).
Clean gap -> midpoint; overlap -> maximise Youden's J. Only 24 images, so treat the numbers as a starting point.

smoke: PASS means the app's preprocessing + model loading reproduce the team's stored vectors
(cosine >= 0.99 per image) and each photo finds its own individual.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app import Gallery, Settings, TorchEmbedder, preprocess

HERE = Path(__file__).resolve().parent


def calibrate_one(emb: np.ndarray, labels: np.ndarray) -> dict:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    S, n = emb @ emb.T, len(labels)
    known = np.array([max(S[q, j] for j in range(n) if j != q and labels[j] == labels[q]) for q in range(n)])
    unknown = np.array([max(S[q, j] for j in range(n) if labels[j] != labels[q]) for q in range(n)])
    if known.min() > unknown.max():
        thr, how = (known.min() + unknown.max()) / 2, "midpoint of the gap"
    else:
        grid = np.linspace(-1, 1, 2001)
        thr = float(grid[int(np.argmax([(known >= t).mean() - (unknown >= t).mean() for t in grid]))])
        how = "max Youden J"
    return {"similarity_threshold": round(float(thr), 4),
            "source": f"model_checks.py calibrate ({how}, leave-one-identity-out, n={n})",
            "known_best_similarity_min": round(float(known.min()), 4),
            "unknown_best_similarity_max": round(float(unknown.max()), 4),
            "known_accept_rate": round(float((known >= thr).mean()), 4),
            "unknown_false_accept_rate": round(float((unknown >= thr).mean()), 4)}


def calibrate() -> None:
    out = {}
    for profile in ("baseline", "finetuned"):
        s = Settings(profile=profile)
        labels = pd.read_csv(s.index_csv).identity.astype(str).values
        out[profile] = calibrate_one(np.load(s.embeddings_npy), labels)
        print(profile, json.dumps(out[profile], indent=2))
    (HERE / "thresholds.json").write_text(json.dumps(out, indent=2))
    print("wrote thresholds.json")


def smoke(profile: str) -> None:
    s = Settings(profile=profile)
    gal, index = Gallery.load(s.embeddings_npy, s.index_csv), pd.read_csv(s.index_csv)
    thr = json.loads((HERE / "thresholds.json").read_text())[profile]["similarity_threshold"]
    emb = TorchEmbedder(profile, s.weights_path)
    emb.load()
    ok_vec = ok_id = 0
    for i, row in index.iterrows():
        v = emb.embed(preprocess(Image.open(s.sample_dir / row.file)))
        cos = float(v @ gal.embeddings[i])
        keep = np.arange(len(index)) != i                       # hide the photo itself
        top = Gallery(gal.embeddings[keep], index[keep]).search(v, 1)[0]
        ok_vec += cos >= 0.99
        ok_id += top.identity == row.identity and top.similarity >= thr
        print(f"{row.file:55s} cos-to-stored={cos:.4f}  top1={top.identity} ({top.similarity:.3f})")
    n = len(index)
    print(f"\nvectors reproduced: {ok_vec}/{n}   correct known-matches: {ok_id}/{n}")
    print("PASS" if ok_vec == n and ok_id == n else "CHECK THE DIFFERENCES ABOVE")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "calibrate":
        calibrate()
    elif len(sys.argv) >= 2 and sys.argv[1] == "smoke":
        smoke(sys.argv[2] if len(sys.argv) > 2 else "baseline")
    else:
        print(__doc__)
