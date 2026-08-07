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

## Full experiment reproduction

Run all commands below from the repository root. The primary experiment uses
the v1 dataset, split, and audio-feature schema for every model. The v2 genre
experiment is supplementary and its absolute metric values are not directly
comparable with the v1 results.

### 1. Environment setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install matplotlib
```

The raw datasets are not included in the repository. Place them at these exact
paths before continuing:

```text
data/raw/spotify_tracks_dataset.csv
data/raw/Music Listening Data (~500k Users)/music.duckdb
```

Run the unit and integration tests:

```bash
python3 -m unittest discover -s tests -v
```

### 2. Build the primary v1 dataset

Run the data-preparation commands in this order:

```bash
python3 scripts/build_databases.py
python3 scripts/match_tracks.py
python3 scripts/cluster_song_features.py

python3 scripts/filter_feature_graph.py \
  --min-user-items 5 \
  --min-item-users 2 \
  --dataset-version feature_graph_u5_i2_v1

python3 scripts/split_feature_graph.py \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --seed 42 \
  --min-evaluation-items 20 \
  --validation-fraction 0.1 \
  --test-fraction 0.1

python3 scripts/build_item_feature_matrix.py \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1
```

These commands create the v1 DuckDB files under `data/databases/` and the
versioned audio feature matrix under `artifacts/features/`.

### 3. Primary v1 validation experiments

Validation data are used for hyperparameter selection. Do not pass a test-data
unlock flag to any validation command.

#### Baselines

```bash
python3 scripts/run_baselines.py \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1 \
  --evaluation-split validation \
  --k 10 20 \
  --seed 42 \
  --output-version baselines_eval20_validation_v1
```

Output: `artifacts/baselines/baselines_eval20_validation_v1.json`.

#### Item-KNN

```bash
python3 scripts/run_cf.py \
  --model item-knn \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1 \
  --data-db-path data/databases/integration.duckdb \
  --alpha 1 10 40 \
  --neighbours 50 100 200 \
  --weighting cosine bm25 \
  --min-cooccurrence 2 5 \
  --k 10 20 \
  --seed 42 \
  --output-version item-knn_validation_v1
```

Output: `artifacts/cf/item-knn_validation_v1.json`.

#### Implicit ALS

The full ALS grid is computationally expensive and may take several hours.

```bash
python3 scripts/run_cf.py \
  --model als \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1 \
  --data-db-path data/databases/integration.duckdb \
  --alpha 1 10 40 \
  --factors 32 64 128 \
  --regularization 0.01 0.1 \
  --iterations 10 20 \
  --k 10 20 \
  --seed 42 \
  --output-version als_validation_v1
```

Output: `artifacts/cf/als_validation_v1.json`.

#### Single-profile CBM

```bash
python3 scripts/run_cbm.py \
  --dataset_version feature_graph_u5_i2_v1 \
  --split_version feature_split_u5_i2_eval20_seed42_v1 \
  --feature_schema_version feature_matrix_audio_v1 \
  --data_db_path data/databases/integration.duckdb \
  --cbm_output_dir artifacts/cbm \
  --ks 10 20 \
  --alphas 0.7 0.8 0.9 0.95
```

Output: `artifacts/cbm/cbm_eval20_validation_v1.json`.

#### Multi-interest CBM

```bash
python3 scripts/run_multicbm.py \
  --dataset_version feature_graph_u5_i2_v1 \
  --split_version feature_split_u5_i2_eval20_seed42_v1 \
  --feature_schema_version feature_matrix_audio_v1 \
  --data_db_path data/databases/integration.duckdb \
  --multicbm_output_dir artifacts/multicbm \
  --multicbm_output_version multicbm_eval20_validation_v3 \
  --ks 10 20 \
  --alphas 0.9 \
  --global_weights 0.1 0.2 0.3 \
  --k_for_kmeans 8 16
```

Output: `artifacts/multicbm/multicbm_eval20_validation_v3.json`. This
experiment is also computationally expensive because it scores the full item
catalog for every validation user and configuration.

#### Hybrid Item-KNN + CBM

```bash
python3 scripts/run_hybrid.py \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1 \
  --data_db_path data/databases/integration.duckdb \
  --alpha 1.0 \
  --neighbours 200 \
  --weighting bm25 \
  --min-cooccurrence 2 \
  --k 10 20 \
  --cf_weight 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 \
  --output-version hybrid_validation_v1
```

Output: `artifacts/hybrid/hybrid_validation_v1.json`.

### 4. Locked v1 final test

Run the test commands only after all hyperparameters have been selected and
frozen using validation data.

```bash
python3 scripts/run_baselines.py \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1 \
  --evaluation-split test \
  --allow-test \
  --k 10 20 \
  --seed 42 \
  --output-version baselines_eval20_test_v1

python3 scripts/run_model_tests.py \
  --allow-test \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1 \
  --data_db_path data/databases/integration.duckdb \
  --output-version test_v1 \
  --output-dir artifacts/test2
```

The final outputs are:

```text
artifacts/baselines/baselines_eval20_test_v1.json
artifacts/test2/test_v1.json
artifacts/test2/test_metrics_plot10.jpg
artifacts/test2/test_metrics_plot20.jpg
```

If `artifacts/test2/test_v1.json` already exists, `run_model_tests.py` loads the
existing result instead of rerunning the models. Move the old result to a backup
location before attempting a clean reproduction.

### 5. Supplementary v2 genre experiment

The v2 experiment adds genre to the item definition and therefore changes item
IDs, interactions, filtering, splits, and the candidate catalog. Treat these
results as a separate supplementary experiment rather than a controlled direct
comparison with v1.

Build v2 in this order:

```bash
python3 scripts/data_for_genre/build_databases_genre.py
python3 scripts/data_for_genre/match_tracks_genre.py
python3 scripts/data_for_genre/cluster_song_features_genre.py

python3 scripts/data_for_genre/filter_feature_graph_genre.py \
  --min-user-items 5 \
  --min-item-users 2 \
  --dataset-version feature_graph_u5_i2_v2 \
  --source_run_id feature_cluster_v2

python3 scripts/data_for_genre/split_feature_graph_genre.py \
  --dataset-version feature_graph_u5_i2_v2 \
  --split-version feature_split_u5_i2_eval20_seed42_v2 \
  --seed 42 \
  --min-evaluation-items 20 \
  --validation-fraction 0.1 \
  --test-fraction 0.1

python3 scripts/data_for_genre/build_item_feature_matrix_genre.py \
  --dataset-version feature_graph_u5_i2_v2 \
  --split-version feature_split_u5_i2_eval20_seed42_v2 \
  --feature-schema-version feature_matrix_audio_genre_v1
```

Run the supplementary v2 CBM experiments by explicitly overriding every data
version and path:

```bash
python3 scripts/run_cbm.py \
  --dataset_version feature_graph_u5_i2_v2 \
  --split_version feature_split_u5_i2_eval20_seed42_v2 \
  --feature_schema_version feature_matrix_audio_genre_v1 \
  --data_db_path data/databases2/integration.duckdb \
  --cbm_output_dir artifacts2/cbm \
  --ks 10 20 \
  --alphas 0.7 0.8 0.9 0.95

python3 scripts/run_multicbm.py \
  --dataset_version feature_graph_u5_i2_v2 \
  --split_version feature_split_u5_i2_eval20_seed42_v2 \
  --feature_schema_version feature_matrix_audio_genre_v1 \
  --data_db_path data/databases2/integration.duckdb \
  --multicbm_output_dir artifacts2/multicbm2 \
  --multicbm_output_version multicbm_eval20_validation_genre_v1 \
  --ks 10 20 \
  --alphas 0.9 \
  --global_weights 0.1 0.2 0.3 \
  --k_for_kmeans 8 16
```

The supplementary outputs are:

```text
artifacts2/cbm/cbm_eval20_validation_v1.json
artifacts2/multicbm2/multicbm_eval20_validation_genre_v1.json
```

### 6. Generate validation figures

After all expected validation JSON files exist, generate the figures with:

```bash
python3 scripts/plot_validations.py
```

Figures are written under `artifacts/figures/`. The combined figure displays
the v1 and v2 models in separately labelled panels; the primary summary panel
must contain only the aligned v1 results.
