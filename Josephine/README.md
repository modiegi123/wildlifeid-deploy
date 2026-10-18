# Part 4 - Gallery & Evaluation (Josephine)

Two independent pieces: an admin app that lets you add new individuals to the gallery (so
the system isn't stuck with only the 8 frozen sample identities forever), and a script that
measures how accurate the whole pipeline actually is against real ground-truth labels.

## 1. Gallery admin (`app.py`)

Add/list known individuals. Reuses Siphesihle's `Settings`, `TorchEmbedder`, `preprocess`
and `validate_image` (Part 3) so validation and preprocessing stay identical across parts -
see `_load_siphesihle_app()` at the top of `app.py` for how that's wired up (both files are
named `app.py`, so it's loaded under a distinct module name rather than a plain import).

New individuals are embedded under **both** profiles (baseline + fine-tuned) at add-time and
written to `gallery_additions/` - a small on-disk store shared with Siphesihle's app:
```
gallery_additions/
  index.csv                 # row,file,identity,species
  images/<uuid>.jpg
  baseline/embeddings.npy   # [M,1536], row-aligned to index.csv
  finetuned/embeddings.npy  # [M,1536], row-aligned to index.csv
```
`Siphesihle/app.py` has one small addition for this (`Settings.additions_dir` +
`Gallery.with_additions()`): when her app builds its default gallery, it folds in whatever is
in `gallery_additions/` for the active profile. So anything added here shows up as a possible
match the next time someone uploads a photo there - no need to re-run anything, and no import
of this folder's code from hers (file-based, same convention Part 3 already uses to read
Part 1/Part 2's artifacts).

### Run
```
pip install -r requirements.txt
python app.py                                   # http://127.0.0.1:5001/admin
```
`GET /api/gallery` -> `{individuals: [{identity, species, n_images, source, sample_image_url}], total}`
`POST /api/gallery` (fields `identity`, `species`, one or more `files`) -> `201 {identity, species, n_images}`
`GET /api/health`. Errors: `{"error": {"code", "message"}}` (same shape as Part 3).

Tests: `python -m pytest -q test_app.py` (fake embedders, no model download, isolated
`tmp_path` gallery so repo state is never touched).

## 2. Evaluation (`evaluate.py`)

Leave-one-out on Mbali's frozen 24-image / 8-identity sample set only (admin additions have
no ground truth, so they're excluded). For each profile (baseline, finetuned), re-embeds
every sample photo with the real model - the same code path as Part 3's `/api/results` - and
measures:
- **top-1 / top-3 identity accuracy** - does the right individual come back?
- **pair recall** - Mbali froze 1 query + 2 photos per identity that should come back as the
  best matches. This checks whether those exact other two photos rank as the top-2 *photos*
  by raw similarity, not just whether the identity label is right.
- **decision accuracy** - `decide()` at the real calibrated threshold from
  `Siphesihle/thresholds.json` (on a closed leave-one-out set the correct decision is always
  "match", since the identity exists elsewhere in the gallery).

```
python evaluate.py                # both profiles -> evaluation_results.json
python evaluate.py baseline       # one profile
```

See `EVALUATION.md` for the write-up with the actual numbers.

## It reads (does not own)
Part 1: `../embeddings/{embeddings.npy,index.csv}` and `../data/sample/` images (via
Siphesihle's `Settings`).
Part 2: `../model_weights.pt` (via the same `Settings`).
Part 3: `../Siphesihle/app.py` (imported for its `Settings`/`Gallery`/`TorchEmbedder`/
`preprocess`/`validate_image`/`decide`) and `../Siphesihle/thresholds.json`.

## It owns
`gallery_additions/` (created at runtime - not committed; starts empty for everyone) and
everything in this folder.
