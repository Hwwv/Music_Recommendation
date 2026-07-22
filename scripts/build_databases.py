#!/usr/bin/env python3
"""Build cleaned DuckDB databases from immutable project source data.

The script is intentionally idempotent: it writes each database to a temporary
path, verifies it, and only then replaces the prior generated database.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - operational error
    raise SystemExit(
        "DuckDB is required. Install it with: "
        "python3 -m pip install --target .tools duckdb==1.4.3"
    ) from exc


RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "databases"
SPOTIFY_CSV = RAW / "spotify_tracks_dataset.csv"
LISTENING_RAW_DB = RAW / "Music Listening Data (~500k Users)" / "music.duckdb"
SPOTIFY_DB = OUT / "spotify.duckdb"
LISTENING_DB = OUT / "listening_clean.duckdb"
INTEGRATION_DB = OUT / "integration.duckdb"


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def fresh_connection(target: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".building")
    if temp.exists():
        temp.unlink()
    return temp, duckdb.connect(str(temp))


def publish(temp: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    temp.replace(target)


def build_spotify() -> None:
    temp, con = fresh_connection(SPOTIFY_DB)
    con.execute("PRAGMA threads=4")
    con.execute(
        f"""
        CREATE TABLE spotify_tracks_staging AS
        SELECT
            column00::INTEGER AS source_row_id,
            trim(track_id)::VARCHAR AS track_id,
            trim(artists)::VARCHAR AS artists,
            trim(album_name)::VARCHAR AS album_name,
            trim(track_name)::VARCHAR AS track_name,
            popularity::SMALLINT AS popularity,
            duration_ms::INTEGER AS duration_ms,
            explicit::BOOLEAN AS explicit,
            danceability::DOUBLE AS danceability,
            energy::DOUBLE AS energy,
            key::SMALLINT AS key,
            loudness::DOUBLE AS loudness,
            mode::SMALLINT AS mode,
            speechiness::DOUBLE AS speechiness,
            acousticness::DOUBLE AS acousticness,
            instrumentalness::DOUBLE AS instrumentalness,
            liveness::DOUBLE AS liveness,
            valence::DOUBLE AS valence,
            tempo::DOUBLE AS tempo,
            time_signature::SMALLINT AS time_signature,
            trim(track_genre)::VARCHAR AS track_genre
        FROM read_csv(
            '{sql_path(SPOTIFY_CSV)}',
            header = true,
            auto_detect = true,
            all_varchar = false
        )
        """
    )
    con.execute(
        """
        CREATE TABLE spotify_rejected_rows AS
        SELECT *,
            CASE
                WHEN track_id IS NULL OR track_id = '' THEN 'missing_track_id'
                WHEN track_name IS NULL OR track_name = '' THEN 'missing_track_name'
                WHEN artists IS NULL OR artists = '' THEN 'missing_artists'
                ELSE 'other'
            END AS rejection_reason
        FROM spotify_tracks_staging
        WHERE track_id IS NULL OR track_id = ''
           OR track_name IS NULL OR track_name = ''
           OR artists IS NULL OR artists = ''
        """
    )
    con.execute(
        """
        CREATE TABLE spotify_feature_issues AS
        SELECT source_row_id, track_id,
            CASE
                WHEN popularity NOT BETWEEN 0 AND 100 THEN 'invalid_popularity'
                WHEN duration_ms <= 0 THEN 'invalid_duration'
                WHEN danceability NOT BETWEEN 0 AND 1 OR energy NOT BETWEEN 0 AND 1
                  OR speechiness NOT BETWEEN 0 AND 1 OR acousticness NOT BETWEEN 0 AND 1
                  OR instrumentalness NOT BETWEEN 0 AND 1 OR liveness NOT BETWEEN 0 AND 1
                  OR valence NOT BETWEEN 0 AND 1 THEN 'invalid_unit_feature'
                WHEN key NOT BETWEEN -1 AND 11 THEN 'invalid_key'
                WHEN mode NOT IN (0, 1) THEN 'invalid_mode'
                WHEN tempo <= 0 THEN 'missing_or_invalid_tempo'
                WHEN time_signature <= 0 THEN 'invalid_time_signature'
                ELSE 'other'
            END AS issue_reason
        FROM spotify_tracks_staging
        WHERE popularity NOT BETWEEN 0 AND 100 OR duration_ms <= 0
           OR danceability NOT BETWEEN 0 AND 1 OR energy NOT BETWEEN 0 AND 1
           OR speechiness NOT BETWEEN 0 AND 1 OR acousticness NOT BETWEEN 0 AND 1
           OR instrumentalness NOT BETWEEN 0 AND 1 OR liveness NOT BETWEEN 0 AND 1
           OR valence NOT BETWEEN 0 AND 1 OR key NOT BETWEEN -1 AND 11
           OR mode NOT IN (0, 1) OR tempo <= 0 OR time_signature <= 0
        """
    )
    con.execute(
        """
        CREATE TABLE spotify_duplicate_audit AS
        SELECT
            track_id,
            count(*) AS source_rows,
            count(DISTINCT track_name) AS distinct_titles,
            count(DISTINCT artists) AS distinct_artist_strings,
            count(DISTINCT track_genre) AS distinct_genres,
            count(DISTINCT struct_pack(
                duration_ms := duration_ms, danceability := danceability,
                energy := energy, key := key, loudness := loudness, mode := mode,
                speechiness := speechiness, acousticness := acousticness,
                instrumentalness := instrumentalness, liveness := liveness,
                valence := valence, tempo := tempo,
                time_signature := time_signature
            )) AS distinct_feature_vectors
        FROM spotify_tracks_staging
        WHERE track_id NOT IN (SELECT track_id FROM spotify_rejected_rows)
        GROUP BY track_id
        HAVING count(*) > 1
        """
    )
    con.execute(
        """
        CREATE TABLE spotify_tracks AS
        WITH valid AS (
            SELECT s.*,
                lower(trim(regexp_replace(track_name, '[^\\p{L}\\p{N}]+', ' ', 'g'))) AS track_name_norm,
                lower(trim(regexp_replace(artists, '[^\\p{L}\\p{N};]+', ' ', 'g'))) AS artists_norm,
                row_number() OVER (
                    PARTITION BY track_id
                    ORDER BY popularity DESC, track_genre, source_row_id
                ) AS choice
            FROM spotify_tracks_staging s
            WHERE source_row_id NOT IN (SELECT source_row_id FROM spotify_rejected_rows)
        )
        SELECT
            track_id, artists, album_name, track_name,
            track_name_norm, artists_norm,
            popularity, duration_ms, explicit,
            danceability, energy, key, loudness, mode, speechiness,
            acousticness, instrumentalness, liveness, valence,
            CASE WHEN tempo > 0 THEN tempo ELSE NULL END AS tempo,
            time_signature
        FROM valid
        WHERE choice = 1
        """
    )
    con.execute("ALTER TABLE spotify_tracks ADD PRIMARY KEY (track_id)")
    con.execute(
        """
        CREATE TABLE spotify_track_genres AS
        SELECT DISTINCT s.track_id, s.track_genre
        FROM spotify_tracks_staging s
        JOIN spotify_tracks t USING (track_id)
        WHERE s.track_genre IS NOT NULL AND s.track_genre <> ''
        """
    )
    con.execute(
        """
        CREATE TABLE spotify_track_artists AS
        SELECT DISTINCT
            t.track_id,
            trim(a.artist_name) AS artist_name,
            lower(trim(regexp_replace(a.artist_name, '[^\\p{L}\\p{N}]+', ' ', 'g'))) AS artist_name_norm,
            a.ordinality::SMALLINT AS artist_order
        FROM spotify_tracks t,
        UNNEST(string_split(t.artists, ';')) WITH ORDINALITY AS a(artist_name, ordinality)
        WHERE trim(a.artist_name) <> ''
        """
    )
    con.execute(
        """
        CREATE TABLE data_quality_summary AS
        SELECT 'spotify_source_rows' AS metric, count(*)::BIGINT AS metric_value FROM spotify_tracks_staging
        UNION ALL SELECT 'spotify_rejected_rows', count(*) FROM spotify_rejected_rows
        UNION ALL SELECT 'spotify_feature_issue_rows', count(*) FROM spotify_feature_issues
        UNION ALL SELECT 'spotify_clean_tracks', count(*) FROM spotify_tracks
        UNION ALL SELECT 'spotify_duplicate_track_ids', count(*) FROM spotify_duplicate_audit
        UNION ALL SELECT 'spotify_feature_conflict_ids', count(*) FROM spotify_duplicate_audit WHERE distinct_feature_vectors > 1
        """
    )
    assert con.sql("SELECT count(*) FROM spotify_tracks").fetchone()[0] > 0
    con.execute("CHECKPOINT")
    con.close()
    publish(temp, SPOTIFY_DB)


def build_listening() -> None:
    temp, con = fresh_connection(LISTENING_DB)
    con.execute("PRAGMA threads=4")
    con.execute(f"ATTACH '{sql_path(LISTENING_RAW_DB)}' AS raw (READ_ONLY)")
    con.execute(
        """
        CREATE TABLE users AS
        SELECT
            user_id::INTEGER AS user_id,
            nullif(trim(country), '')::VARCHAR AS country,
            total_scrobbles::BIGINT AS total_scrobbles
        FROM raw.users
        WHERE user_id IS NOT NULL AND total_scrobbles IS NOT NULL AND total_scrobbles >= 0
        QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY total_scrobbles DESC) = 1
        """
    )
    con.execute("ALTER TABLE users ADD PRIMARY KEY (user_id)")
    con.execute(
        """
        CREATE TABLE listening_rejected_rows AS
        SELECT *,
            CASE
                WHEN user_id IS NULL THEN 'missing_user_id'
                WHEN track_name IS NULL OR trim(track_name) = '' THEN 'missing_track_name'
                WHEN artist_name IS NULL OR trim(artist_name) = '' THEN 'missing_artist_name'
                WHEN playcount IS NULL OR playcount <= 0 THEN 'invalid_playcount'
                WHEN rank IS NULL OR rank <= 0 THEN 'invalid_rank'
                WHEN user_id NOT IN (SELECT user_id FROM users) THEN 'unknown_user'
                ELSE 'other'
            END AS rejection_reason
        FROM raw.user_top_tracks
        WHERE user_id IS NULL
           OR track_name IS NULL OR trim(track_name) = ''
           OR artist_name IS NULL OR trim(artist_name) = ''
           OR playcount IS NULL OR playcount <= 0
           OR rank IS NULL OR rank <= 0
           OR user_id NOT IN (SELECT user_id FROM users)
        """
    )
    con.execute(
        """
        CREATE TABLE user_top_tracks AS
        WITH normalized AS (
            SELECT
                user_id::INTEGER AS user_id,
                rank::SMALLINT AS rank,
                trim(track_name)::VARCHAR AS track_name,
                trim(artist_name)::VARCHAR AS artist_name,
                lower(trim(regexp_replace(track_name, '[^\\p{L}\\p{N}]+', ' ', 'g'))) AS track_name_norm,
                lower(trim(regexp_replace(artist_name, '[^\\p{L}\\p{N}]+', ' ', 'g'))) AS artist_name_norm,
                playcount::BIGINT AS playcount,
                nullif(trim(mbid), '')::VARCHAR AS mbid
            FROM raw.user_top_tracks
            WHERE user_id IS NOT NULL
              AND track_name IS NOT NULL AND trim(track_name) <> ''
              AND artist_name IS NOT NULL AND trim(artist_name) <> ''
              AND playcount IS NOT NULL AND playcount > 0
              AND rank IS NOT NULL AND rank > 0
              AND user_id IN (SELECT user_id FROM users)
        )
        SELECT
            user_id,
            arg_max(track_name, playcount) AS track_name,
            arg_max(artist_name, playcount) AS artist_name,
            track_name_norm,
            artist_name_norm,
            max(playcount)::BIGINT AS playcount,
            min(rank)::SMALLINT AS source_rank,
            arg_max(mbid, playcount) FILTER (WHERE mbid IS NOT NULL) AS mbid,
            count(*)::INTEGER AS merged_source_rows
        FROM normalized
        GROUP BY user_id, track_name_norm, artist_name_norm
        """
    )
    con.execute("CREATE INDEX idx_listening_user ON user_top_tracks(user_id)")
    con.execute(
        """
        CREATE TABLE listening_track_keys AS
        SELECT
            track_name_norm,
            artist_name_norm,
            arg_max(track_name, playcount) AS representative_track_name,
            arg_max(artist_name, playcount) AS representative_artist_name,
            count(*)::BIGINT AS interaction_count,
            count(DISTINCT user_id)::INTEGER AS user_count,
            sum(playcount)::HUGEINT AS total_playcount
        FROM user_top_tracks
        GROUP BY track_name_norm, artist_name_norm
        """
    )
    con.execute(
        """
        CREATE TABLE data_quality_summary AS
        SELECT 'raw_users' AS metric, count(*)::BIGINT AS metric_value FROM raw.users
        UNION ALL SELECT 'clean_users', count(*) FROM users
        UNION ALL SELECT 'raw_track_interactions', count(*) FROM raw.user_top_tracks
        UNION ALL SELECT 'rejected_track_rows', count(*) FROM listening_rejected_rows
        UNION ALL SELECT 'clean_track_interactions', count(*) FROM user_top_tracks
        UNION ALL SELECT 'distinct_listening_track_keys', count(*) FROM listening_track_keys
        UNION ALL SELECT 'users_with_5_plus_tracks', count(*) FROM (SELECT user_id FROM user_top_tracks GROUP BY user_id HAVING count(*) >= 5)
        UNION ALL SELECT 'users_with_10_plus_tracks', count(*) FROM (SELECT user_id FROM user_top_tracks GROUP BY user_id HAVING count(*) >= 10)
        UNION ALL SELECT 'users_with_20_plus_tracks', count(*) FROM (SELECT user_id FROM user_top_tracks GROUP BY user_id HAVING count(*) >= 20)
        """
    )
    assert con.sql("SELECT count(*) FROM users").fetchone()[0] > 0
    assert con.sql("SELECT count(*) FROM user_top_tracks").fetchone()[0] > 0
    con.execute("DETACH raw")
    con.execute("CHECKPOINT")
    con.close()
    publish(temp, LISTENING_DB)


def build_integration() -> None:
    temp, con = fresh_connection(INTEGRATION_DB)
    con.execute(f"ATTACH '{sql_path(SPOTIFY_DB)}' AS spotify (READ_ONLY)")
    con.execute(f"ATTACH '{sql_path(LISTENING_DB)}' AS listening (READ_ONLY)")
    con.execute("CREATE TABLE project_users AS SELECT user_id FROM listening.users")
    con.execute("ALTER TABLE project_users ADD PRIMARY KEY (user_id)")
    con.execute(
        """
        CREATE TABLE project_tracks AS
        SELECT
            track_id, track_name, artists, popularity, duration_ms, explicit,
            danceability, energy, key, loudness, mode, speechiness,
            acousticness, instrumentalness, liveness, valence, tempo,
            time_signature
        FROM spotify.spotify_tracks
        """
    )
    con.execute("ALTER TABLE project_tracks ADD PRIMARY KEY (track_id)")
    con.execute(
        """
        CREATE TABLE track_match_candidates (
            track_name_norm VARCHAR NOT NULL,
            artist_name_norm VARCHAR NOT NULL,
            spotify_track_id VARCHAR NOT NULL REFERENCES project_tracks(track_id),
            match_tier VARCHAR NOT NULL,
            title_similarity DOUBLE,
            artist_similarity DOUBLE,
            PRIMARY KEY (track_name_norm, artist_name_norm, spotify_track_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE track_match_decisions (
            track_name_norm VARCHAR NOT NULL,
            artist_name_norm VARCHAR NOT NULL,
            candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0),
            decision VARCHAR NOT NULL CHECK (decision IN ('accepted', 'rejected', 'ambiguous')),
            accepted_spotify_track_id VARCHAR REFERENCES project_tracks(track_id),
            decision_reason VARCHAR,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            ,PRIMARY KEY (track_name_norm, artist_name_norm)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE track_crosswalk (
            track_name_norm VARCHAR NOT NULL,
            artist_name_norm VARCHAR NOT NULL,
            spotify_track_id VARCHAR NOT NULL REFERENCES project_tracks(track_id),
            match_tier VARCHAR NOT NULL,
            title_similarity DOUBLE NOT NULL,
            artist_similarity DOUBLE NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            PRIMARY KEY (track_name_norm, artist_name_norm)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE interactions_integrated (
            user_id INTEGER NOT NULL REFERENCES project_users(user_id),
            spotify_track_id VARCHAR NOT NULL REFERENCES project_tracks(track_id),
            playcount_raw BIGINT NOT NULL CHECK (playcount_raw > 0),
            preference SMALLINT NOT NULL DEFAULT 1 CHECK (preference = 1),
            confidence_log DOUBLE NOT NULL CHECK (confidence_log >= 1),
            source_rank SMALLINT,
            match_tier VARCHAR NOT NULL,
            merged_listening_key_count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (user_id, spotify_track_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE dataset_splits (
            split_version VARCHAR NOT NULL,
            seed INTEGER NOT NULL,
            user_id INTEGER NOT NULL REFERENCES project_users(user_id),
            spotify_track_id VARCHAR NOT NULL REFERENCES project_tracks(track_id),
            split VARCHAR NOT NULL CHECK (split IN ('train', 'validation', 'test')),
            PRIMARY KEY (split_version, user_id, spotify_track_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE integration_audit (
            run_id VARCHAR NOT NULL,
            metric VARCHAR NOT NULL,
            value DOUBLE NOT NULL,
            recorded_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            PRIMARY KEY (run_id, metric)
        )
        """
    )
    con.execute("DETACH spotify")
    con.execute("DETACH listening")
    con.execute("CHECKPOINT")
    con.close()
    publish(temp, INTEGRATION_DB)


def summarize() -> None:
    for path in (SPOTIFY_DB, LISTENING_DB, INTEGRATION_DB):
        con = duckdb.connect(str(path), read_only=True)
        tables = [row[0] for row in con.sql("SHOW TABLES").fetchall()]
        print(f"\n{path.relative_to(ROOT)} ({path.stat().st_size / 1024**2:.1f} MiB)")
        for table in tables:
            count = con.sql(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            print(f"  {table}: {count:,}")
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("spotify", "listening", "integration", "all"), default="all")
    args = parser.parse_args()
    if args.only in ("spotify", "all"):
        print("Building Spotify database...")
        build_spotify()
    if args.only in ("listening", "all"):
        print("Building listening database...")
        build_listening()
    if args.only in ("integration", "all"):
        if not SPOTIFY_DB.exists() or not LISTENING_DB.exists():
            raise SystemExit("Build Spotify and listening databases before integration.")
        print("Building integration schema...")
        build_integration()
    summarize()


if __name__ == "__main__":
    main()
