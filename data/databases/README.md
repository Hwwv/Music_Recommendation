# Generated databases

These files are generated from immutable sources under `data/raw/` by
`scripts/build_databases.py`. Database binaries are intentionally ignored by
version control; schemas and transformations live in the build script.

## What “schemas and transformations live in the build script” means

The `.duckdb` files contain the current materialized tables, but they are build
outputs, similar to compiled files. They are large, binary, and unsuitable for
reviewing changes line by line. The reproducible definition of those files is
the readable SQL executed by `scripts/build_databases.py`.

In that script, **schema** means the code that defines table names, columns,
types, keys, and constraints. Examples include:

```python
con.execute("ALTER TABLE spotify_tracks ADD PRIMARY KEY (track_id)")
```

and:

```sql
CREATE TABLE interactions_integrated (
    user_id INTEGER NOT NULL REFERENCES project_users(user_id),
    spotify_track_id VARCHAR NOT NULL REFERENCES project_tracks(track_id),
    playcount_raw BIGINT NOT NULL CHECK (playcount_raw > 0),
    ...
    PRIMARY KEY (user_id, spotify_track_id)
)
```

**Transformation** means the SQL that converts a source table into a cleaned or
derived table. For example, the Spotify build:

- trims identity text;
- casts popularity and duration into numeric types;
- rejects rows missing track identity;
- converts a non-positive tempo to `NULL` while recording the issue;
- selects one canonical row per duplicate `track_id`;
- splits semicolon-delimited collaborations into `spotify_track_artists`.

The listening build similarly validates users and interactions, creates
normalized title/artist keys, and merges duplicate normalized interactions by
taking the maximum play count and best rank.

Therefore, changing a table by manually editing a generated database is not a
durable project change: the edit disappears at the next rebuild. To make a
durable change, update the corresponding `CREATE TABLE ... AS SELECT` or
`CREATE TABLE (...)` statement in `scripts/build_databases.py`, rebuild, and
inspect the audit counts. This provides a traceable route:

```text
immutable raw data
       +
versioned build rules
       ↓
reproducible generated databases
```

## Databases

### `spotify.duckdb`

- `spotify_tracks_staging`: all 114,000 typed source rows
- `spotify_tracks`: one canonical row per valid Spotify `track_id`
- `spotify_track_artists`: collaboration-aware artist bridge
- `spotify_track_genres`: many-to-many track/genre bridge
- `spotify_duplicate_audit`: duplicate and feature-consistency checks
- `spotify_rejected_rows`: rows lacking matchable identity
- `spotify_feature_issues`: invalid/missing feature values retained for audit
- `data_quality_summary`: build counts

### `listening_clean.duckdb`

- `users`: one validated row per user
- `user_top_tracks`: validated and normalized song interactions
- `listening_track_keys`: unique normalized title/artist keys for efficient matching
- `listening_rejected_rows`: invalid source rows and reasons
- `data_quality_summary`: build and cohort counts

The large supplied `data/raw/Music Listening Data (~500k Users)/music.duckdb`
remains the immutable source. The cleaned database intentionally materializes
only users and song-level interactions needed by the current project.

### `integration.duckdb`

- `project_users` and `project_tracks`: local key dimensions
- `track_match_candidates`: exact-match Spotify candidates for listening keys
- `track_match_decisions`: accepted, ambiguous, and unmatched decisions
- `track_crosswalk`: accepted listening-to-Spotify mappings
- `interactions_integrated`: matched, user-level interactions
- `spotify_feature_clusters`: exact title/primary-artist/acoustic fingerprints
- `spotify_feature_cluster_members`: Spotify IDs belonging to each fingerprint
- `listening_feature_candidates`: listening keys mapped to distinct feature clusters
- `listening_feature_decisions` and `listening_feature_crosswalk`: cluster-level decisions
- `feature_interactions_integrated`: user interactions keyed by feature cluster
- `dataset_splits`: versioned train/validation/test assignments
- `integration_audit`: matching and integration metrics

The database build creates these matching-dependent tables empty. Running
`scripts/match_tracks.py` then populates the conservative `exact_v1` baseline.
It accepts only a unique exact normalized title plus exact artist match;
multiple Spotify candidates remain ambiguous, and zero-candidate keys remain
unmatched.

## Runnable examples

Run all three read-only examples from the project root:

```bash
PYTHONPATH=.tools python3 scripts/database_examples.py
```

### Example 1 — Spotify database

Find highly popular, high-energy tracks and include all their genre labels:

```sql
SELECT
    t.track_id,
    t.track_name,
    t.artists,
    t.popularity,
    t.energy,
    t.danceability,
    list(g.track_genre ORDER BY g.track_genre) AS genres
FROM spotify_tracks AS t
LEFT JOIN spotify_track_genres AS g USING (track_id)
WHERE t.popularity >= 80 AND t.energy >= 0.80
GROUP BY ALL
ORDER BY t.popularity DESC, t.track_name
LIMIT 5;
```

This demonstrates that `spotify_tracks` is the one-row-per-item table while
genres live in a separate many-to-many bridge.

### Example 2 — Listening database

Retrieve one user’s cleaned top tracks:

```sql
SELECT
    user_id,
    source_rank,
    track_name,
    artist_name,
    playcount,
    1 + ln(1 + playcount) AS example_confidence
FROM user_top_tracks
WHERE user_id = 1
ORDER BY source_rank
LIMIT 10;
```

This demonstrates that `playcount` is preserved as raw implicit feedback. The
confidence expression is an example only; its `alpha` multiplier must later be
tuned on validation data.

### Example 3 — Integration database

Inspect the populated dimensions and current matching progress:

```sql
SELECT
    (SELECT count(*) FROM project_users) AS available_users,
    (SELECT count(*) FROM project_tracks) AS available_spotify_tracks,
    (SELECT count(*) FROM track_crosswalk) AS reviewed_crosswalk_rows,
    (SELECT count(*) FROM interactions_integrated) AS integrated_interactions,
    (SELECT count(*) FROM spotify_feature_clusters) AS feature_clusters,
    (SELECT count(*) FROM listening_feature_crosswalk) AS feature_crosswalk_rows,
    (SELECT count(*) FROM feature_interactions_integrated) AS feature_interactions;
```

After running only `build_databases.py --only integration`, the final two values
are zero. After running `match_tracks.py`, they contain the accepted exact-match
crosswalk and integrated interactions.

```text
available_users = 476451
available_spotify_tracks = 89740
reviewed_crosswalk_rows = 35788
integrated_interactions = 2937852
feature_clusters = 83851
feature_crosswalk_rows = 37343
feature_interactions = 3125625
```

After matching begins, an accepted crosswalk row must reference an existing
`project_tracks.track_id`, and an integrated interaction must reference both an
existing project user and project track. Those foreign keys prevent orphaned
records.

### Inspect any schema directly

Inside DuckDB, use:

```sql
SHOW TABLES;
DESCRIBE spotify_tracks;
SELECT * FROM duckdb_constraints();
```

The first command lists tables, the second shows columns and types, and the
third shows primary keys, foreign keys, `NOT NULL`, and `CHECK` constraints.

## Rebuild

From the project root:

```bash
PYTHONPATH=.tools python3 scripts/build_databases.py
```

Build one layer:

```bash
PYTHONPATH=.tools python3 scripts/build_databases.py --only spotify
PYTHONPATH=.tools python3 scripts/build_databases.py --only listening
PYTHONPATH=.tools python3 scripts/build_databases.py --only integration
```

Populate the conservative exact-match integration after the databases exist:

```bash
PYTHONPATH=.tools python3 scripts/match_tracks.py
```

Then collapse feature-equivalent Spotify IDs and build cluster-level interactions:

```bash
PYTHONPATH=.tools python3 scripts/cluster_song_features.py
```

Build the versioned model-ready interaction graph by iteratively retaining users
with at least 5 clusters and clusters heard by at least 2 users:

```bash
PYTHONPATH=.tools python3 scripts/filter_feature_graph.py \
  --min-user-items 5 \
  --min-item-users 2 \
  --dataset-version feature_graph_u5_i2_v1
```

The pruning repeats until both degree constraints hold simultaneously. Results
are stored in `feature_graph_datasets` and `feature_graph_interactions`; the
associated counts and minimum final degrees are stored in `integration_audit`
under the dataset version. Dataset versions are immutable: rerunning the same
version and thresholds reuses it, while changed thresholds require a new name.

Create the primary deterministic split. Users with at least 20 retained items
receive approximately 10% validation and 10% test items; sparser users remain
entirely in training:

```bash
PYTHONPATH=.tools python3 scripts/split_feature_graph.py \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --seed 42 \
  --min-evaluation-items 20
```

Split metadata is stored in `feature_split_datasets`, assignments in
`feature_dataset_splits`, and leakage/coverage counts in `integration_audit`.

Build the versioned train-fitted acoustic item feature matrix:

```bash
PYTHONPATH=.tools python3 scripts/build_item_feature_matrix.py \
  --dataset-version feature_graph_u5_i2_v1 \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --feature-schema-version feature_matrix_audio_v1
```

The Parquet matrix and JSON metadata are written under `artifacts/features/`.
The schema registry and artifact checksum are stored in `item_feature_schemas`.

Run the three primary validation baselines and shared top-k evaluation:

```bash
PYTHONPATH=.tools:src python3 scripts/run_baselines.py \
  --split-version feature_split_u5_i2_eval20_seed42_v1 \
  --evaluation-split validation \
  --k 10 20 \
  --seed 42 \
  --output-version baselines_eval20_validation_v1
```

Test evaluation is locked by default and requires the explicit `--allow-test`
flag after model selection is frozen.

Export feature-level review samples:

```bash
PYTHONPATH=.tools python3 scripts/export_feature_cluster_review.py
```

Rebuilding `integration.duckdb` resets its matching-dependent tables. Run
`match_tracks.py` again afterward to reproduce `exact_v1`.
Then run `cluster_song_features.py` to reproduce `feature_cluster_v1`.

Each build writes to a temporary database, verifies non-empty canonical tables,
checkpoints the file, and only then replaces the previous generated database.
