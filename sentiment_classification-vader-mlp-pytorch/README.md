# sentiment_classification-vader-mlp-pytorch

## 1. Task summary

- **Task:** 3-class sentiment classification (negative / neutral / positive).
- **Dataset:** Embedded 60-review corpus (20 per class), tiled 4× → 240 reviews. Movie + product + restaurant + miscellaneous factual reviews.
- **Models:** VADER (rule-based, NLTK's `SentimentIntensityAnalyzer`) vs `nnx.NNModel` MLP (spaCy lemmatize → 100-token BoW → 32-hidden classifier).
- **Framework:** PyTorch (via [`nnx`](../nnx)) + NLTK + spaCy + sklearn.

## 2. Why this exists

Sentiment classification has two famously-good recipes:

- **VADER** (Hutto & Gilbert, 2014) — hand-tuned lexicon + grammar rules. No training. Ships with NLTK. Strong baseline; in many production settings it beats supervised neural models on out-of-distribution text.
- **Supervised neural** — BoW + MLP, trained on labeled examples. Wins on in-distribution text; loses on domain shift.

This notebook runs both on the same embedded review corpus and reports the head-to-head accuracy. The §6.3 prose makes the production-relevant point: always include VADER as a baseline — hand-tuned lexicons are *much* harder to beat than the supervised-ML literature acknowledges.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/sentiment_classification-vader-mlp-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — both recipes, dataset, libraries.
- §2 Environment & Setup — imports (lazy `nltk.download('vader_lexicon')`), hyperparameters (`VOCAB_SIZE=100`, `HIDDEN_DIMS=[32]`, `N_EPOCHS=60`, `COMPOUND_POS_THRESHOLD=0.05` / `_NEG=-0.05` per VADER paper), `nnx.set_seed(0)`.
- §3 Data — embedded 60-review corpus tiled 4×, 80/20 stratified split, spaCy tokenize + train-only BoW vocabulary + L2-normalized featurizer (same recipe as the sibling AG-News notebook).
- §4 Model — VADER contract + small `FeedFwdNN` classifier.
- §5 Training — VADER predicts in one call (no training); neural MLP trained for 60 epochs.
- §6 Evaluation & Results — accuracy table + per-class `classification_report` for both recipes; side-by-side confusion-matrix heatmaps; §6.3 discussion of when each recipe wins in production.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier A** (cheap, ~10 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch` — autograd + tensors.
- `nnx` (the submodule) — `FeedFwdNN`, `NNModel`, `NNTrainParams`, `set_seed`.
- `nltk` — `SentimentIntensityAnalyzer`. The `vader_lexicon` is downloaded lazily by §2.1 if not already present (cached after first run).
- `spacy` — tokenizer + lemmatizer (`en_core_web_sm`, same as `text_classification-agnews-spacy-mlp-pytorch/`).
- `scikit-learn` — split + accuracy + classification_report + confusion_matrix.
- `numpy`, `matplotlib`, `prettytable`.

`nltk` is not currently in `requirements.txt` — add `nltk` to root requirements before merging this task (or rely on the lazy download path documented in §2.1).

## 6. Known issues

- **Embedded corpus is tiny (60 unique reviews, 240 after tiling).** Real sentiment benchmarks (IMDB 50k, Amazon Reviews) are much larger. Absolute accuracy numbers are not directly comparable. The *relative* ordering (VADER ≈ neural at this scale) is the pedagogical point.
- **`nltk` lazy download.** The first run downloads `vader_lexicon` (~125 KB) via `nltk.download`. CI containers may not have this preloaded; the notebook's try/except handles it but the first run takes a few extra seconds.
- **VADER's compound thresholds (±0.05) are the original-paper defaults.** Domain-specific tuning (e.g., compound > 0.2 for more conservative positive classification) helps on noisy real-world text; we use defaults for reproducibility.
- **Neutral detection is the weak spot for both recipes.** Embedded "neutral" reviews are factual statements with no polarity words (release dates, store hours, package weights). VADER calls these neutral by default — good. The neural MLP can learn this if the train split contains enough factual-statement vocabulary; with 16 neutral train samples, it sometimes overfits to specific neutral keywords.
- **`nltk` is not in `requirements.txt`.** This will need to be added before the task can run on a fresh `pip install -r requirements.txt`. The lazy `nltk.download('vader_lexicon')` won't help if the package itself isn't installed.
