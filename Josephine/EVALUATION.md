# Evaluation

## What this measures, and why there are two layers

Keightly's `comparison_metrics.csv` and `fine_tuned_eval.json` already show the fine-tuned
model separates embeddings better than the baseline (mean cosine similarity for the *same*
individual rose from 0.77 to 0.25... distance, i.e. similarity fell for same-identity pairs
too — see the correction below under "embedding quality"). That's a statement about the
**embeddings**. It says nothing about whether the deployed app actually makes the right
*decision* when someone uploads a photo, because that also depends on the open-set threshold.

Siphesihle found Keightly's own calibration constant (tau = 0.929, a *distance*, i.e.
similarity 0.071) wrongly accepts ~40% of unknown animals as known individuals in her own
leave-one-identity-out test, and recalibrated to similarity thresholds of **0.1985**
(baseline) and **0.1734** (fine-tuned) - the ones actually wired into the app via
`thresholds.json`. Those are the thresholds used below, because they're what the deployed
system uses, not the notebook's figure.

So this evaluation runs the **real, deployed pipeline end to end** - real model, real
preprocessing, real `Gallery.search()`, real `decide()` - rather than re-deriving accuracy
from the raw vectors, to check that "the fine-tuned model looks better on paper" actually
holds up once it's wired into the app people use.

## Method

Leave-one-out over Mbali's frozen 24-image / 8-identity sample set (`data/sample/`). Each of
the 24 photos is held out, re-embedded with the real model, and matched against the
remaining 23. Run separately for both profiles. Reproduce with `python evaluate.py`.

Three things are measured per photo:
- **Identity accuracy (top-1 / top-3)** - does `Gallery.search()` return the correct
  individual as the best (or one of the top-3) match?
- **Pair recall** - Mbali froze 1 query + 2 photos per identity specifically so the other two
  should come back as the best matches. This checks whether those exact two photos are the
  top-2 ranked *photos* by raw similarity - a stricter, more literal test of the dataset's
  design than identity accuracy alone (which aggregates to best-photo-per-individual and can
  hide a photo-level miss).
- **Decision accuracy** - `decide()` at the real calibrated threshold. On this closed
  leave-one-out set every identity exists elsewhere in the gallery, so the correct decision
  is always "match"; this measures how often the open-set threshold agrees.

## Results

| Metric | Baseline | Fine-tuned |
|---|---|---|
| Top-1 identity accuracy | 1.000 | 1.000 |
| Top-3 identity accuracy | 1.000 | 1.000 |
| **Pair recall** | **1.000** | **0.917** |
| Decision accuracy (real threshold) | 1.000 (thr 0.1985) | 1.000 (thr 0.1734) |

(`evaluation_results.json` has the full per-image breakdown.)

## Reading these results honestly

Both profiles hit 100% on identity accuracy and decision accuracy - which is a ceiling
effect, not proof the fine-tuned model is flawless: with only 24 images across 8 identities,
every individual is visually distinct enough (different species, in most cases) that
identity-level matching was already easy for the baseline. These two metrics don't have
room left to show a difference on this sample set.

**Pair recall is the one metric that did show a difference, and it went the other way than
expected.** Baseline got all 24/24; fine-tuned got 22/24, missing both `DogFaceNet_208`
photos - each failed to rank the *other* same-identity photo in its top-2 by raw similarity,
even though the identity-aggregated top-1 was still correct both times (`evaluation_results.json`
-> `finetuned.per_image`, rows for `003_DogFaceNet_208.jpg` / `005_DogFaceNet_208.jpg`).
Fine-tuning was done on zebra/leopard/lion/giraffe - dogs were not in that subset - so this
looks like a plausible case of the fine-tuned model spreading photos of an out-of-distribution
species slightly further apart from each other than the pretrained baseline did, even while
still getting the coarser identity match right.

**Embedding quality** (Keightly's `comparison_metrics.csv`): mean cosine similarity for the
*same* individual actually **fell** from 0.77 (baseline) to 0.25 (fine-tuned), while mean
similarity for *different* individuals rose from 0.008 to 0.98. Both numbers moving away from
each other is the right direction for separability in principle, but a same-identity
similarity of only 0.25 is itself unusually low - combined with the pair-recall dip above, it
suggests the fine-tuned model's notion of "distance" shifted in a way that isn't uniformly an
improvement, not that it strictly dominates the baseline on every axis.

## Bottom line

On this 24-image sample, both the baseline and fine-tuned models drive the deployed app to
100% correct decisions once given app's actual recalibrated thresholds (not Keightly's
notebook tau) - so neither model is shown to be clearly better for the real upload/results
flow yet. The one place a difference shows up (pair recall) slightly favours the baseline,
on a species that wasn't part of the fine-tuning subset. The honest conclusion is that 24
images / 8 identities isn't enough to separate these two models with confidence either way;
a larger, more diverse sample set (more identities, and including species the fine-tuning
subset covers vs. doesn't) is needed before picking one model over the other for production.
