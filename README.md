# CSC311 Hybrid Music Recommender

Reproducible implementation of the project proposed in the shared **CSC311 Project Proposal**.

## Research questions

1. Does combining collaborative filtering (CF) and content-based modelling (CBM) improve top-k recommendation quality?
2. Does play-count confidence improve listening prediction?
3. Can multi-interest content profiles increase acoustic diversity and exposure to less-popular songs without sacrificing too much relevance?

## Repository map

```text
src/music_recommender/   reusable models, data utilities, and metrics
scripts/run_demo.py      end-to-end smoke experiment on deterministic toy data
tests/                   dependency-free unit and integration tests
data/raw/                local Kaggle downloads (ignored by git)
data/processed/          generated matched tables (ignored by git)
artifacts/               experiment outputs (ignored by git)
```

## Quick start

The initial implementation uses only Python's standard library so it runs before the large datasets are downloaded.

```bash
python3 scripts/run_demo.py
python3 -m unittest discover -s tests -v
```

## Sparse collaborative filtering

The full-data CF implementations use binary preference and log-play-count
confidence, `c_ui = 1 + alpha * log(1 + playcount)`. Item-KNN calculates sparse
item similarities in blocks and retains only top-K neighbours; implicit ALS
optimizes the confidence-weighted objective over observed and unobserved pairs.
Tune on validation only:

```bash
PYTHONPATH=.tools python3 scripts/run_cf.py --model item-knn \
  --alpha 1 10 40 --neighbours 50 100 200 \
  --weighting cosine bm25 --min-cooccurrence 2 5

PYTHONPATH=.tools python3 scripts/run_cf.py --model als \
  --alpha 1 10 40 --factors 32 64 128 \
  --regularization 0.01 0.1 --iterations 10 20
```

The commands save ranked configurations under `artifacts/cf/`. The ALS method
follows Hu, Koren, and Volinsky, *Collaborative Filtering for Implicit Feedback
Datasets* (ICDM 2008), included at `reference/cf.pdf`. Test data remain locked
during model selection.

## Unified experiment loader

`MusicDataLoader.load_experiment("validation")` returns one version-checked
`ExperimentData` bundle containing typed training interactions, held-out truth,
item features, the canonical catalog, and per-user seen items. CF, content, and
hybrid runners should consume this bundle instead of issuing independent
database queries. The record and feature-mapping methods do not require pandas;
the older DataFrame methods remain available for analysis notebooks.

## Dataset contract

Place the Kaggle exports under `data/raw/` (they are intentionally not committed):

- Spotify Tracks: one row per track, including `track_name`, `artists`, and numeric acoustic features.
- Music Listening Dataset: one row per user-track interaction, including `user_id`, track/artist names, and `play_count`.

Before modelling, track names and artists must be normalized, joined, and audited. Ambiguous many-to-one matches should be removed rather than silently assigned. All train/test splits are performed per user before fitting models.

## Experiment matrix

| Family | Variant | Purpose |
|---|---|---|
| Non-personalized | Popularity | Sanity-check baseline |
| CF | Item KNN | Neighbourhood baseline |
| CF | Matrix factorization | Latent implicit-feedback model |
| CBM | Single profile | Acoustic-feature baseline |
| CBM | Multi-interest profile | Avoid averaging distinct tastes |
| Hybrid | CF + CBM | Test complementarity |

Primary relevance metrics are Recall@K and NDCG@K. Beyond-accuracy metrics are catalog coverage, intra-list diversity, novelty, and long-tail share. Every final comparison should report uncertainty across users or repeated seeds.
