# Part 3 - Upload & Results (Siphesihle)

Upload a photo -> it is validated and resized/normalised -> MegaDescriptor embeds it -> it is compared with the
gallery -> the page shows **known individual** or **possible new individual**.

## Files (all yours)
| File | What it is |
|---|---|
| `app.py` | the whole backend: settings, upload validation, preprocessing, gallery search, model, upload store, routes |
| `templates/index.html` | upload page (drag & drop, preview, errors) |
| `templates/results.html` | results page (verdict, top matches, threshold bar) |
| `static/app.css` | styling |
| `thresholds.json` | open-set similarity threshold per model |
| `model_checks.py` | `calibrate` thresholds / `smoke` test the real model |
| `test_app.py` | 20 automated tests |
| `requirements.txt` | packages |

## Run
```
pip install -r requirements.txt
python app.py                                   # http://127.0.0.1:5000  (baseline; downloads ~800 MB from Hugging Face once)
```
Fine-tuned model: copy `model_weights.pt` into `../Part2_Keightly/`, then
`set WILDLIFEID_PROFILE=finetuned` (PowerShell: `$env:WILDLIFEID_PROFILE="finetuned"`) and `python app.py`.
Tests: `python -m pytest -q test_app.py`   Real-model check: `python model_checks.py smoke baseline`

## It reads (does not own)
Part 1: `../Part1_Mbali/embeddings/{embeddings.npy,index.csv}` and `data/sample/` images.
Part 2: `../Part2_Keightly/fine_tuned_embeddings.npy` and `model_weights.pt`.

## API
`POST /api/upload` (field `file`) -> `201 {upload_id, results_url, ...}` | `GET /api/results/<id>` -> decision, threshold,
top matches | `GET /api/health`. Errors: `{"error": {"code", "message"}}`.

## Threshold note
Keightly's `tau = 0.929` is a distance = similarity 0.071, which wrongly accepts ~40% of unknown animals in a
leave-one-identity-out test. This app uses calibrated similarity thresholds (baseline 0.1985, fine-tuned 0.1734).
Override with `WILDLIFEID_SIM_THRESHOLD=0.0707`. Only 8 individuals were available, so recalibrate on more data.
