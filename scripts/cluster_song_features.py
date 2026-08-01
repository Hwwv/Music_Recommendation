#!/usr/bin/env python3
"""Cluster duplicate Spotify IDs by exact acoustic fingerprint and remap matches."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

try:
    import duckdb
except ImportError as exc:
    raise SystemExit("DuckDB is required; see data/databases/README.md") from exc

DB = ROOT / "data" / "databases"
INTEGRATION = DB / "integration.duckdb"
SPOTIFY = DB / "spotify.duckdb"
LISTENING = DB / "listening_clean.duckdb"


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main() -> None:
    con = duckdb.connect(str(INTEGRATION))
    con.execute("PRAGMA threads=4")
    con.execute(f"ATTACH '{sql_path(SPOTIFY)}' AS spotify (READ_ONLY)")
    con.execute(f"ATTACH '{sql_path(LISTENING)}' AS listening (READ_ONLY)")

    con.execute("DELETE FROM item_feature_schemas")
    con.execute("DELETE FROM feature_dataset_splits")
    con.execute("DELETE FROM feature_split_datasets")
    # Downstream model-ready graphs become stale whenever feature interactions
    # are rebuilt. Delete children before their version registry rows.
    con.execute("DELETE FROM feature_graph_interactions")
    con.execute("DELETE FROM feature_graph_datasets")
    con.execute("DELETE FROM feature_interactions_integrated")
    con.execute("DELETE FROM listening_feature_crosswalk")
    con.execute("DELETE FROM listening_feature_decisions")
    con.execute("DELETE FROM listening_feature_candidates")
    con.execute("DELETE FROM spotify_feature_cluster_members")
    con.execute("DELETE FROM spotify_feature_clusters")
    con.execute("DELETE FROM integration_audit WHERE run_id = 'feature_cluster_v1'")

    # Each Spotify track receives one stable exact fingerprint. Popularity,
    # album, genre, and track_id are deliberately excluded because they describe
    # catalog placement rather than the audio content. Explicit is retained so
    # clean and explicit releases cannot be merged automatically.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE feature_members_staging AS
        WITH primary_artists AS (
            SELECT track_id, artist_name_norm AS primary_artist_norm
            FROM spotify.spotify_track_artists
            WHERE artist_order = 1
        ), fingerprinted AS (
            SELECT
                t.*,
                coalesce(a.primary_artist_norm, '') AS primary_artist_norm,
                sha256(to_json(struct_pack(
                    track_name_norm := t.track_name_norm,
                    primary_artist_norm := coalesce(a.primary_artist_norm, ''),
                    duration_ms := t.duration_ms,
                    explicit := t.explicit,
                    danceability := t.danceability,
                    energy := t.energy,
                    key := t.key,
                    loudness := t.loudness,
                    mode := t.mode,
                    speechiness := t.speechiness,
                    acousticness := t.acousticness,
                    instrumentalness := t.instrumentalness,
                    liveness := t.liveness,
                    valence := t.valence,
                    tempo := t.tempo,
                    time_signature := t.time_signature
                ))) AS feature_cluster_id
            FROM spotify.spotify_tracks t
            LEFT JOIN primary_artists a USING (track_id)
        )
        SELECT
            *,
            row_number() OVER (
                PARTITION BY feature_cluster_id
                ORDER BY popularity DESC NULLS LAST, track_id
            ) AS canonical_order,
            count(*) OVER (PARTITION BY feature_cluster_id)::INTEGER AS member_count
        FROM fingerprinted
        """
    )

    con.execute(
        """
        INSERT INTO spotify_feature_clusters
        SELECT
            feature_cluster_id,
            track_id AS canonical_track_id,
            track_name_norm,
            primary_artist_norm,
            member_count,
            popularity AS canonical_popularity,
            duration_ms,
            explicit,
            danceability,
            energy,
            key,
            loudness,
            mode,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo,
            time_signature
        FROM feature_members_staging
        WHERE canonical_order = 1
        """
    )

    con.execute(
        """
        INSERT INTO spotify_feature_cluster_members
        SELECT
            feature_cluster_id,
            track_id,
            album_name,
            popularity,
            canonical_order = 1
        FROM feature_members_staging
        """
    )

    # Compress exact Spotify-ID candidates into distinct feature candidates.
    con.execute(
        """
        INSERT INTO listening_feature_candidates
        SELECT
            c.track_name_norm,
            c.artist_name_norm,
            m.feature_cluster_id,
            count(DISTINCT c.spotify_track_id)::INTEGER AS spotify_id_count,
            'exact_title_artist_feature_cluster' AS match_tier
        FROM track_match_candidates c
        JOIN spotify_feature_cluster_members m
          ON c.spotify_track_id = m.spotify_track_id
        GROUP BY c.track_name_norm, c.artist_name_norm, m.feature_cluster_id
        """
    )

    con.execute(
        """
        INSERT INTO listening_feature_decisions (
            track_name_norm, artist_name_norm, feature_cluster_count,
            spotify_candidate_count, decision, accepted_feature_cluster_id,
            decision_reason
        )
        WITH counts AS (
            SELECT
                k.track_name_norm,
                k.artist_name_norm,
                count(DISTINCT f.feature_cluster_id)::INTEGER AS cluster_count,
                coalesce(sum(f.spotify_id_count), 0)::INTEGER AS spotify_count,
                min(f.feature_cluster_id) AS only_cluster
            FROM listening.listening_track_keys k
            LEFT JOIN listening_feature_candidates f
              USING (track_name_norm, artist_name_norm)
            GROUP BY k.track_name_norm, k.artist_name_norm
        )
        SELECT
            track_name_norm,
            artist_name_norm,
            cluster_count,
            spotify_count,
            CASE
                WHEN cluster_count = 1 THEN 'accepted'
                WHEN cluster_count = 0 THEN 'rejected'
                ELSE 'ambiguous'
            END,
            CASE WHEN cluster_count = 1 THEN only_cluster END,
            CASE
                WHEN cluster_count = 1 AND spotify_count = 1
                    THEN 'one exact Spotify candidate and one feature cluster'
                WHEN cluster_count = 1 AND spotify_count > 1
                    THEN 'multiple exact Spotify IDs collapse to one feature-equivalent cluster'
                WHEN cluster_count = 0
                    THEN 'no exact normalized title and artist candidate'
                ELSE 'multiple distinct acoustic feature clusters remain'
            END
        FROM counts
        """
    )

    con.execute(
        """
        INSERT INTO listening_feature_crosswalk
        SELECT
            d.track_name_norm,
            d.artist_name_norm,
            d.accepted_feature_cluster_id,
            c.canonical_track_id,
            f.spotify_id_count,
            f.match_tier,
            current_timestamp
        FROM listening_feature_decisions d
        JOIN listening_feature_candidates f
          ON d.track_name_norm = f.track_name_norm
         AND d.artist_name_norm = f.artist_name_norm
         AND d.accepted_feature_cluster_id = f.feature_cluster_id
        JOIN spotify_feature_clusters c
          ON d.accepted_feature_cluster_id = c.feature_cluster_id
        WHERE d.decision = 'accepted'
        """
    )

    con.execute(
        """
        INSERT INTO feature_interactions_integrated
        WITH matched AS (
            SELECT
                i.user_id,
                x.feature_cluster_id,
                x.canonical_track_id,
                i.playcount,
                i.source_rank,
                i.track_name_norm,
                i.artist_name_norm
            FROM listening.user_top_tracks i
            JOIN listening_feature_crosswalk x
              USING (track_name_norm, artist_name_norm)
        )
        SELECT
            user_id,
            feature_cluster_id,
            min(canonical_track_id) AS canonical_track_id,
            max(playcount)::BIGINT AS playcount_raw,
            1::SMALLINT AS preference,
            1 + ln(1 + max(playcount)) AS confidence_log,
            min(source_rank)::SMALLINT AS source_rank,
            count(DISTINCT struct_pack(
                track_name_norm := track_name_norm,
                artist_name_norm := artist_name_norm
            ))::INTEGER AS merged_listening_key_count
        FROM matched
        GROUP BY user_id, feature_cluster_id
        """
    )

    con.execute(
        """
        INSERT INTO integration_audit (run_id, metric, value)
        SELECT 'feature_cluster_v1', 'spotify_tracks', count(*) FROM project_tracks
        UNION ALL SELECT 'feature_cluster_v1', 'spotify_feature_clusters', count(*) FROM spotify_feature_clusters
        UNION ALL SELECT 'feature_cluster_v1', 'multi_id_feature_clusters', count(*) FROM spotify_feature_clusters WHERE member_count > 1
        UNION ALL SELECT 'feature_cluster_v1', 'accepted_listening_keys', count(*) FROM listening_feature_decisions WHERE decision = 'accepted'
        UNION ALL SELECT 'feature_cluster_v1', 'accepted_multi_id_keys', count(*) FROM listening_feature_crosswalk WHERE spotify_id_count > 1
        UNION ALL SELECT 'feature_cluster_v1', 'ambiguous_listening_keys', count(*) FROM listening_feature_decisions WHERE decision = 'ambiguous'
        UNION ALL SELECT 'feature_cluster_v1', 'unmatched_listening_keys', count(*) FROM listening_feature_decisions WHERE decision = 'rejected'
        UNION ALL SELECT 'feature_cluster_v1', 'feature_interactions', count(*) FROM feature_interactions_integrated
        UNION ALL SELECT 'feature_cluster_v1', 'feature_users', count(DISTINCT user_id) FROM feature_interactions_integrated
        UNION ALL SELECT 'feature_cluster_v1', 'represented_feature_clusters', count(DISTINCT feature_cluster_id) FROM feature_interactions_integrated
        UNION ALL SELECT 'feature_cluster_v1', 'users_with_5_plus_features', count(*) FROM (SELECT user_id FROM feature_interactions_integrated GROUP BY user_id HAVING count(*) >= 5)
        UNION ALL SELECT 'feature_cluster_v1', 'users_with_10_plus_features', count(*) FROM (SELECT user_id FROM feature_interactions_integrated GROUP BY user_id HAVING count(*) >= 10)
        UNION ALL SELECT 'feature_cluster_v1', 'users_with_20_plus_features', count(*) FROM (SELECT user_id FROM feature_interactions_integrated GROUP BY user_id HAVING count(*) >= 20)
        """
    )

    # Integrity checks.
    assert con.sql("SELECT count(*) FROM spotify_feature_cluster_members").fetchone()[0] == con.sql("SELECT count(*) FROM project_tracks").fetchone()[0]
    assert con.sql("SELECT count(DISTINCT spotify_track_id) FROM spotify_feature_cluster_members").fetchone()[0] == con.sql("SELECT count(*) FROM project_tracks").fetchone()[0]
    assert con.sql("SELECT count(*) FROM listening_feature_crosswalk").fetchone()[0] == con.sql("SELECT count(*) FROM listening_feature_decisions WHERE decision='accepted'").fetchone()[0]
    assert con.sql("SELECT count(*) FROM feature_interactions_integrated WHERE playcount_raw <= 0 OR confidence_log < 1").fetchone()[0] == 0
    con.execute("CHECKPOINT")

    print("Feature-cluster integration audit")
    for metric, value in con.sql(
        "SELECT metric, value FROM integration_audit WHERE run_id='feature_cluster_v1' ORDER BY metric"
    ).fetchall():
        print(f"  {metric}: {int(value):,}")

    con.execute("DETACH spotify")
    con.execute("DETACH listening")
    con.close()


if __name__ == "__main__":
    main()
