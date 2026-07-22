# Dataset cleaning and integration guide

This guide is specific to the raw files currently stored in `data/raw/`. Raw files are inputs and should never be edited in place. Every cleaning step should write a new table or file under `data/processed/`, and every row-count change should be recorded.

## 1. Files and roles

### Spotify track features

`raw/spotify_tracks_dataset.csv`

- 114,000 data rows
- One unused CSV index column plus:
  `track_id`, `artists`, `album_name`, `track_name`, `popularity`,
  `duration_ms`, `explicit`, 11 acoustic/audio fields, and `track_genre`
- Supplies item identifiers, acoustic features, genre, and Spotify popularity.
- It does **not** contain user interactions.

The same Spotify `track_id` may occur in more than one genre row. Treat `track_id` as the candidate item identity and audit duplicate rows before deduplicating.

### Music Listening Data (~500k Users)

The folder contains 476,451 user rows and approximately 23.8 million rows in each top-list table.

| File | Use in this project |
|---|---|
| `users.csv` | Optional user metadata and plausibility checks |
| `user_top_tracks.csv` | **Primary CF interaction table**: user, track, artist, rank, play count |
| `user_top_artists.csv` | Optional artist-preference features; do not treat as track interactions |
| `user_top_albums.csv` | Optional album-preference features; do not treat as track interactions |
| `music.duckdb` | Existing database copy of the four CSV tables |

`mbid` in the listening tables is a MusicBrainz identifier, not a Spotify `track_id`; the two IDs cannot be joined directly.

## 2. Processing principles

1. Use DuckDB or another out-of-core engine. Do not load a 1.6 GB interaction CSV into pandas in one operation.
2. Preserve raw values alongside normalized matching values.
3. Never fuzzy-match directly into the final table. Generate candidates, score them, inspect samples, and accept only unambiguous matches.
4. Split train/validation/test **after** integration and filtering, but before model fitting or popularity calculations.
5. Report results for the population that survives matching. This population is a selected subset and may not represent all 476,451 users.

The provided `load.sql` should not be run unchanged: its paths contain a leading space and refer to a `music.duckdb/` directory rather than the current CSV locations. The existing `music.duckdb` can be queried directly; otherwise create a new database with corrected paths.

## 3. Recommended output layers

Keep the following logical layers, preferably as DuckDB tables plus optional Parquet exports:

```text
stg_spotify_tracks       typed Spotify rows, not yet deduplicated
spotify_tracks_clean     one row per Spotify track_id
spotify_track_artists    one normalized artist per track_id
stg_listening_tracks     typed listening rows, raw text retained
listening_tracks_clean   valid, deduplicated user-track-name interactions
track_match_candidates   possible Spotify matches with evidence
track_match_decisions    one accepted/ambiguous/unmatched decision per listening key
track_crosswalk          one accepted listening key -> one Spotify track_id
interactions_integrated  final user_id, track_id, playcount table
item_features            one feature vector per retained track_id
```

For large intermediate outputs, prefer Parquet over CSV because it preserves types and is much faster to scan.

## 4. Clean the Spotify dataset

### 4.1 Ingest and type

- Drop the unnamed first CSV index column; it is not a feature.
- Read `track_id`, `artists`, `album_name`, `track_name`, and `track_genre` as strings.
- Parse `explicit` as Boolean.
- Parse the following as numeric:
  `popularity`, `duration_ms`, `danceability`, `energy`, `key`, `loudness`,
  `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`,
  `valence`, `tempo`, and `time_signature`.
- Reject rows missing `track_id`, `track_name`, or `artists` from the matchable item table, but retain their counts in the audit.

### 4.2 Validate ranges

Flag rather than silently repair values outside these expected ranges:

| Field | Expected check |
|---|---|
| popularity | integer 0–100 |
| duration_ms | positive; inspect extreme short/long values |
| danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence | 0–1 |
| key | -1 or 0–11 |
| mode | 0 or 1 |
| tempo | positive |
| time_signature | small positive integer; inspect unusual values |

Do not remove a track solely for low popularity: long-tail exposure is one of the project outcomes.

### 4.3 Deduplicate Spotify tracks

Group by `track_id` and calculate:

- number of rows;
- number of distinct titles, artist strings, albums, and genres;
- minimum and maximum of each acoustic feature.

If duplicate rows have the same identity and audio features but different genres, retain one item row and store the genres as a separate set/list. If audio features conflict for the same `track_id`, quarantine the track for inspection instead of arbitrarily averaging.

### 4.4 Normalize identity text

Create matching-only fields while retaining originals:

- Unicode normalization and case folding;
- normalize whitespace and punctuation;
- normalize `&` consistently;
- create both a conservative title and an optional relaxed title;
- split Spotify's semicolon-separated `artists` field into individual artists;
- store a normalized artist set.

Recommended fields:

```text
track_name_raw
track_name_norm
track_name_relaxed
artists_raw
primary_artist_norm
artist_set_norm
```

The conservative title should preserve meaningful version text such as `live`, `remix`, `acoustic`, and `radio edit`. The relaxed title may remove bracketed version text, but it must only be used to propose lower-confidence candidates.

### 4.5 Prepare model features

The first content model should use:

```text
danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo
```

- Fit imputation and scaling parameters on the training items only.
- Standardize continuous fields.
- Consider cyclical encoding for `key`; treat `mode` separately.
- Do not use `track_id`, raw names, CSV row number, or user-independent Spotify popularity as acoustic dimensions.
- Keep `popularity` separately for long-tail and novelty analysis.
- Use genre only in a clearly labelled metadata-enhanced CBM experiment, not the pure-acoustic baseline.

## 5. Clean the listening dataset

### 5.1 Begin with `user_top_tracks.csv`

Required columns are:

```text
user_id, rank, track_name, artist_name, playcount, mbid
```

Cast `user_id`, `rank`, and `playcount` to integers. Preserve `mbid` as nullable text. Remove or quarantine rows with:

- null user, track, or artist;
- blank normalized track or artist;
- non-positive play count;
- rank outside the apparent top-list range;
- impossible duplicated rank for the same user, unless inspection shows a legitimate tie convention.

### 5.2 Check table invariants

Produce an audit table containing:

- total rows and distinct users;
- min/median/max rows per user;
- distinct `(track_name, artist_name)` pairs;
- missing and blank counts per column;
- duplicate `(user_id, track_name, artist_name)` counts;
- play-count and rank quantiles;
- percentage of users represented in `users.csv`;
- whether ranks are unique and ordered within each user;
- whether `playcount` generally decreases as rank increases.

Do not assume `rank` is a rating. It is ordering information. The main implicit signal is that the interaction occurred; `playcount` supplies confidence.

### 5.3 Deduplicate interactions

Normalize title and artist using the same conservative functions used for Spotify. Define the listening identity key as:

```text
(track_name_norm, artist_name_norm)
```

For repeated rows belonging to the same user and normalized key:

- retain one interaction;
- use the maximum play count unless the source documentation proves rows represent additive periods;
- preserve the best/smallest rank;
- retain all observed non-null MBIDs for audit.

Do not aggregate different tracks merely because their titles match; artist identity is mandatory.

### 5.4 Use user metadata cautiously

From `users.csv`, retain `country` and `total_scrobbles` for descriptive analysis. Check whether the sum of the top-track play counts can plausibly exceed `total_scrobbles`; if it does, investigate source definitions before using the latter as a denominator.

Country should not be required for the core CF model. Using it would change the project into a demographic or context-aware recommender experiment.

### 5.5 Transform play counts

Store both raw and transformed values:

```text
playcount_raw
preference = 1
confidence_log = 1 + alpha * ln(1 + playcount_raw)
```

Tune `alpha` on validation data. Compare against an unweighted binary model to answer the proposal's play-count research question. The raw play count should not be treated directly as a 1–5-style rating.

## 6. Integrate listening tracks with Spotify

Integration is an entity-resolution problem because the datasets share names but not identifiers.

### 6.1 Match unique track keys first

Build the set of distinct normalized listening `(title, artist)` keys before matching. Match each unique key once, then join the resulting crosswalk back to all users. This is far cheaper than matching tens of millions of interaction rows independently.

### 6.2 Candidate hierarchy

Use the following ordered rules:

1. **Tier A — exact conservative match**  
   Exact normalized title and exact normalized artist membership; exactly one Spotify `track_id` candidate.
2. **Tier B — exact title, artist-set match**  
   Exact title and at least one exact artist overlap for collaborations; one candidate.
3. **Tier C — relaxed title match**  
   Relaxed title plus exact artist overlap. Accept automatically only if version information is compatible and the candidate is unique.
4. **Tier D — fuzzy candidate, manual/thresholded**  
   High title similarity plus exact or very-high artist similarity. Use only after manually labelling a random candidate sample and choosing thresholds.

Reject automatic matches when:

- more than one Spotify track remains after applying the rule;
- only the title matches;
- artist names conflict;
- version text conflicts (`live` versus studio, remix versus original, acoustic versus original);
- the candidate is a karaoke, tribute, or cover recording without matching artist evidence.

Spotify may contain multiple legitimate `track_id` values for effectively the same recording. For the first experiment, select only when there is a deterministic rule—for example, exact identity plus compatible version and the highest-popularity candidate—and record the alternatives. Otherwise mark the key ambiguous.

### 6.3 Crosswalk fields

The final `track_crosswalk` should contain at least:

```text
listening_track_name_raw
listening_artist_name_raw
track_name_norm
artist_name_norm
spotify_track_id
match_tier
title_similarity
artist_similarity
candidate_count
decision
decision_reason
```

`decision` should be one of `accepted`, `rejected`, or `ambiguous`. Never discard ambiguous cases without counting them.

### 6.4 Validate matching quality

For each tier, manually inspect a reproducible random sample. Include common and rare tracks, punctuation differences, collaborations, non-Latin names, and version suffixes.

Report:

- unique listening keys matched by tier;
- interaction rows and distinct users recovered by tier;
- precision estimate from the labelled audit sample;
- ambiguous and unmatched proportions;
- distribution of Spotify popularity for matched versus unmatched items;
- user-history length before and after matching.

Do **not** choose a matching threshold only because it maximizes the number of eligible users. A false match creates incorrect positives and contaminates both CF and evaluation.

## 7. Construct the integrated modelling tables

Join accepted crosswalk rows back to cleaned listening interactions:

```text
interactions_integrated
-----------------------
user_id
track_id
playcount_raw
preference
confidence_log
source_rank
match_tier
```

Then enforce one row per `(user_id, track_id)`. If multiple listening keys map to the same Spotify track for one user, take the maximum play count by default and flag the collision count.

Create `item_features` with one row per retained `track_id`, scaled acoustic fields, genre metadata, and Spotify popularity.

Only retain interactions whose `track_id` exists in `item_features` for the fair CF/CBM/hybrid comparison. Keep counts from before this alignment so the coverage loss is visible.

## 8. Filtering and experiment cohorts

Do not begin by retaining only users with at least 20 matched tracks. That would throw away useful CF co-occurrence data.

Recommended sequence:

1. Start from all accepted, integrated interactions.
2. Iteratively remove users with fewer than 5 retained tracks and items listened to by fewer than 2 users until stable.
3. Treat the resulting table as the broad CF training graph.
4. Define the main evaluation cohort as users with at least 20 retained tracks.
5. Also report results for history-size bands such as 5–9, 10–19, and 20+.
6. Run sensitivity versions such as user/item thresholds `(5,2)`, `(10,3)`, and `(20,5)`.

The iterative step matters: removing sparse items can make a previously eligible user sparse again.

## 9. Train/validation/test split

Split within each eligible user's history so every evaluated user still has training interactions.

For a user with at least 20 matched items, a reasonable starting point is:

- approximately 80% train;
- approximately 10% validation;
- approximately 10% test;
- at least one item in validation and one in test.

If timestamps are unavailable, use deterministic random splits with several seeds and state that chronological evaluation was impossible. The source contains top tracks rather than a complete listening log, so absence of an interaction is “unknown,” not a confirmed dislike.

Fit all of the following on training data only:

- item/user popularity;
- CF similarities and latent factors;
- feature imputation and scaling;
- multi-interest clusters;
- hybrid normalization parameters.

Tune hyperparameters with validation users/items. Report the untouched test results once the design is fixed.

## 10. Quality-control report required before modelling

Create one reproducible audit report with:

1. raw and cleaned row counts for every source;
2. missingness, duplicates, and type/range violations;
3. Spotify duplicate-track analysis;
4. matching yield and sampled precision by tier;
5. matched interactions, users, and items;
6. user and item degree distributions before and after iterative filtering;
7. matrix density and connected-component sizes;
8. counts of users in each history-size cohort;
9. play-count and popularity distributions;
10. exact split sizes and leakage checks.

The integration is ready for modelling only when:

- every accepted listening key maps to exactly one Spotify `track_id`;
- every final interaction has a corresponding item-feature row;
- no `(user_id, track_id)` appears in more than one split;
- held-out items are absent from that user's training history;
- row-count changes and exclusions are explainable from the audit tables.

## 11. Suggested first implementation order

1. Query or recreate the DuckDB staging tables.
2. Generate the Spotify duplicate and range audit.
3. Generate the listening-track invariant audit.
4. Normalize unique title/artist keys on both sides.
5. Build and inspect Tier A/B matches only.
6. Measure how many users retain 5, 10, and 20 matched tracks.
7. Add Tier C only if its audited precision is acceptable.
8. Freeze `track_crosswalk`, create integrated tables, and run iterative filtering.
9. Freeze deterministic dataset and split versions before training models.

This order gives an early, defensible exact-match baseline and prevents fuzzy matching choices from being driven by downstream model scores.

## 12. Current exact-match baseline (`exact_v1`)

The first implemented matching rule requires:

- exact normalized track title;
- exact normalized listening artist against one artist in the Spotify
  collaboration-aware artist bridge;
- exactly one candidate Spotify `track_id` for automatic acceptance.

Current results are:

| Metric | Count |
|---|---:|
| Listening title/artist keys | 3,075,750 |
| Accepted unique exact keys | 34,976 |
| Ambiguous exact keys | 3,878 |
| Unmatched keys | 3,036,896 |
| Integrated user-track interactions | 2,934,320 |
| Users with at least one exact match | 427,845 |
| Spotify tracks represented | 34,525 |
| Users with at least 5 matches | 249,804 |
| Users with at least 10 matches | 106,026 |
| Users with at least 20 matches | 13,821 |

Fifty-one integrated interactions combine multiple listening title/artist keys
that resolved to the same `(user_id, spotify_track_id)`. These use maximum
play count and minimum rank rather than summing potentially duplicated source
representations. Relaxed or fuzzy matching should not begin until accepted and
ambiguous samples from this baseline are reviewed.
