# WildlifeID

Open-set individual animal re-identification. Upload a photo and the app compares it against a
gallery of known individuals using MegaDescriptor embeddings.

- `/` - upload a photo, get a match or "no confident match"
- `/admin/` - add new individuals to the gallery
- `/pipeline/` - baseline + fine-tuning results report

This deployment runs the lighter `MegaDescriptor-T-224` model (768-d embeddings, separately
calibrated - see `Siphesihle/thresholds.json`) via `onnxruntime` instead of PyTorch, to fit
free-tier hosting memory limits (PyTorch alone needs ~300MB just to import). The gallery starts
with the 8 frozen sample identities; anything added via `/admin/` persists only for this
deployment's lifetime (no persistent disk).
