"""Part 4 (Josephine): evaluation script.

    python evaluate.py                  # both profiles
    python evaluate.py baseline         # one profile only

Leave-one-out on Mbali's frozen 24-image / 8-identity sample set (never admin additions -
those have no ground truth to check against). For each image, re-embeds it with the REAL
model - the same code path as Siphesihle's /api/results - and measures:

  top-1 / top-3 identity accuracy   Gallery.search(), identity-aggregated (best photo per
                                     individual), same as what the results page shows.
  pair recall                       Mbali froze 1 query + 2 "should come back as best
                                     matches" photos per identity. This checks whether those
                                     exact other two photos are the top-2 ranked PHOTOS by
                                     raw similarity - stronger than just the identity label
                                     matching, since it tests the dataset's actual design.
  decision accuracy                 decide() at the real calibrated threshold from
                                     Siphesihle/thresholds.json. On a closed leave-one-out
                                     set the identity always exists elsewhere, so the correct
                                     decision is always "match".

Needs torch + timm installed (same as Siphesihle/model_checks.py smoke); downloads the
baseline weights from Hugging Face once if not already cached.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app import Gallery, Settings, TorchEmbedder, decide, preprocess

HERE = Path(__file__).resolve().parent
SIPHESIHLE_DIR = HERE.parent / "Siphesihle"


def _threshold(profile: str) -> float:
    data = json.loads((SIPHESIHLE_DIR / "thresholds.json").read_text())[profile]
    return float(data["similarity_threshold"])


def evaluate_profile(profile: str) -> dict:
    settings = Settings(profile=profile)
    index = pd.read_csv(settings.index_csv)
    gallery = Gallery.load(settings.embeddings_npy, settings.index_csv)
    threshold = _threshold(profile)
    embedder = TorchEmbedder(profile, settings.weights_path)
    embedder.load()

    n = len(index)
    rows = []
    for i, row in index.iterrows():
        img = Image.open(settings.sample_dir / row.file)
        v = embedder.embed(preprocess(img, settings.input_size))

        keep = np.arange(n) != i
        reduced_index = index[keep].reset_index(drop=True)
        reduced = Gallery(gallery.embeddings[keep], reduced_index)

        top = reduced.search(v, k=3)
        verdict = decide(top, threshold)

        q = v / max(float(np.linalg.norm(v)), 1e-12)
        sims = reduced.embeddings @ q
        ranked = np.argsort(-sims)
        same_identity_rows = np.where(reduced_index.identity.values == row.identity)[0]
        pair_recall = bool(set(same_identity_rows).issubset(set(ranked[: len(same_identity_rows)])))

        rows.append({
            "file": row.file, "identity": row.identity, "species": row.species,
            "top1_identity": top[0].identity if top else None,
            "top1_similarity": round(top[0].similarity, 4) if top else None,
            "top1_correct": bool(top and top[0].identity == row.identity),
            "top3_correct": bool(top and row.identity in {m.identity for m in top}),
            "pair_recall": pair_recall,
            "decision": verdict["decision"],
            "decision_correct": verdict["decision"] == "match",
        })

    df = pd.DataFrame(rows)
    return {
        "profile": profile,
        "n_images": n,
        "n_identities": int(index.identity.nunique()),
        "threshold": threshold,
        "top1_accuracy": round(float(df.top1_correct.mean()), 4),
        "top3_accuracy": round(float(df.top3_correct.mean()), 4),
        "pair_recall": round(float(df.pair_recall.mean()), 4),
        "decision_accuracy": round(float(df.decision_correct.mean()), 4),
        "per_image": rows,
    }


def main(profiles: tuple[str, ...]) -> None:
    results = {}
    for profile in profiles:
        print(f"\n=== {profile} ===")
        result = evaluate_profile(profile)
        results[profile] = result
        print(f"top-1 identity accuracy:  {result['top1_accuracy']:.3f}")
        print(f"top-3 identity accuracy:  {result['top3_accuracy']:.3f}")
        print(f"pair recall:              {result['pair_recall']:.3f}")
        print(f"decision accuracy:        {result['decision_accuracy']:.3f}  "
              f"(threshold {result['threshold']})")
    out = HERE / "evaluation_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("baseline", "finetuned"))
