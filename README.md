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

