#!/usr/bin/env python3
"""Build a conservative exact-match listening-to-Spotify crosswalk."""

from __future__ import annotations

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

DB = ROOT / "data" / "databases2"
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

    # This script owns these derived tables. Re-running it produces the same
    # exact-match baseline from the current cleaned source databases.
    con.execute("DELETE FROM dataset_splits")
    con.execute("DELETE FROM interactions_integrated")
    con.execute("DELETE FROM track_crosswalk")
    con.execute("DELETE FROM track_match_decisions")
    con.execute("DELETE FROM track_match_candidates")
    con.execute("DELETE FROM integration_audit")

    con.execute(
        """
        INSERT INTO track_match_candidates
        SELECT DISTINCT
            l.track_name_norm,
            l.artist_name_norm,
            s.track_id AS spotify_track_id,
            'exact_title_exact_artist' AS match_tier,
            1.0 AS title_similarity,
            1.0 AS artist_similarity
        FROM listening.listening_track_keys AS l
        JOIN spotify.spotify_tracks AS s
          ON l.track_name_norm = s.track_name_norm
        JOIN spotify.spotify_track_artists AS a
          ON s.track_id = a.track_id
         AND l.artist_name_norm = a.artist_name_norm
        WHERE l.track_name_norm <> ''
          AND l.artist_name_norm <> ''
          AND s.track_name_norm <> ''
          AND a.artist_name_norm <> ''
        """
    )

    con.execute(
        """
        INSERT INTO track_match_decisions (
            track_name_norm, artist_name_norm, candidate_count,
            decision, accepted_spotify_track_id, decision_reason
        )
        WITH counts AS (
            SELECT
                l.track_name_norm,
                l.artist_name_norm,
                count(DISTINCT c.spotify_track_id)::INTEGER AS candidate_count,
                min(c.spotify_track_id) AS only_candidate
            FROM listening.listening_track_keys AS l
            LEFT JOIN track_match_candidates AS c
              USING (track_name_norm, artist_name_norm)
            GROUP BY l.track_name_norm, l.artist_name_norm
        )
        SELECT
            track_name_norm,
            artist_name_norm,
            candidate_count,
            CASE
                WHEN candidate_count = 1 THEN 'accepted'
                WHEN candidate_count = 0 THEN 'rejected'
                ELSE 'ambiguous'
            END AS decision,
            CASE WHEN candidate_count = 1 THEN only_candidate END AS accepted_spotify_track_id,
            CASE
                WHEN candidate_count = 1 THEN 'unique exact normalized title and artist match'
                WHEN candidate_count = 0 THEN 'no exact normalized title and artist candidate'
                ELSE 'multiple Spotify track IDs share the exact normalized title and artist'
            END AS decision_reason
        FROM counts
        """
    )

    con.execute(
        """
        INSERT INTO track_crosswalk
        SELECT
            d.track_name_norm,
            d.artist_name_norm,
            d.accepted_spotify_track_id,
            c.match_tier,
            c.title_similarity,
            c.artist_similarity,
            current_timestamp
        FROM track_match_decisions AS d
        JOIN track_match_candidates AS c
          ON d.track_name_norm = c.track_name_norm
         AND d.artist_name_norm = c.artist_name_norm
         AND d.accepted_spotify_track_id = c.spotify_track_id
        WHERE d.decision = 'accepted'
        """
    )

    con.execute(
        """
        INSERT INTO interactions_integrated
        WITH matched AS (
            SELECT
                i.user_id,
                x.spotify_track_id,
                i.playcount,
                i.source_rank,
                x.match_tier,
                i.track_name_norm,
                i.artist_name_norm
            FROM listening.user_top_tracks AS i
            JOIN track_crosswalk AS x
              USING (track_name_norm, artist_name_norm)
        )
        SELECT
            user_id,
            spotify_track_id,
            max(playcount)::BIGINT AS playcount_raw,
            1::SMALLINT AS preference,
            1 + ln(1 + max(playcount)) AS confidence_log,
            min(source_rank)::SMALLINT AS source_rank,
            min(match_tier) AS match_tier,
            count(DISTINCT struct_pack(
                track_name_norm := track_name_norm,
                artist_name_norm := artist_name_norm
            ))::INTEGER AS merged_listening_key_count
        FROM matched
        GROUP BY user_id, spotify_track_id
        """
    )

    con.execute(
        """
        INSERT INTO integration_audit (run_id, metric, value)
        SELECT 'exact_v2', 'listening_track_keys', count(*) FROM listening.listening_track_keys
        UNION ALL SELECT 'exact_v2', 'candidate_rows', count(*) FROM track_match_candidates
        UNION ALL SELECT 'exact_v2', 'accepted_track_keys', count(*) FROM track_match_decisions WHERE decision = 'accepted'
        UNION ALL SELECT 'exact_v2', 'ambiguous_track_keys', count(*) FROM track_match_decisions WHERE decision = 'ambiguous'
        UNION ALL SELECT 'exact_v2', 'unmatched_track_keys', count(*) FROM track_match_decisions WHERE decision = 'rejected'
        UNION ALL SELECT 'exact_v2', 'integrated_interactions', count(*) FROM interactions_integrated
        UNION ALL SELECT 'exact_v2', 'integrated_users', count(DISTINCT user_id) FROM interactions_integrated
        UNION ALL SELECT 'exact_v2', 'integrated_spotify_tracks', count(DISTINCT spotify_track_id) FROM interactions_integrated
        UNION ALL SELECT 'exact_v2', 'users_with_5_plus_matches', count(*) FROM (SELECT user_id FROM interactions_integrated GROUP BY user_id HAVING count(*) >= 5)
        UNION ALL SELECT 'exact_v2', 'users_with_10_plus_matches', count(*) FROM (SELECT user_id FROM interactions_integrated GROUP BY user_id HAVING count(*) >= 10)
        UNION ALL SELECT 'exact_v2', 'users_with_20_plus_matches', count(*) FROM (SELECT user_id FROM interactions_integrated GROUP BY user_id HAVING count(*) >= 20)
        UNION ALL SELECT 'exact_v2', 'interactions_merging_multiple_listening_keys', count(*) FROM interactions_integrated WHERE merged_listening_key_count > 1
        """
    )

    # Constraint and cardinality checks.
    accepted = con.sql("SELECT count(*) FROM track_match_decisions WHERE decision='accepted'").fetchone()[0]
    crosswalk = con.sql("SELECT count(*) FROM track_crosswalk").fetchone()[0]
    assert accepted == crosswalk
    assert con.sql("SELECT count(*) FROM track_crosswalk WHERE spotify_track_id NOT IN (SELECT track_id FROM project_tracks)").fetchone()[0] == 0
    assert con.sql("SELECT count(*) FROM interactions_integrated WHERE playcount_raw <= 0 OR confidence_log < 1").fetchone()[0] == 0
    con.execute("CHECKPOINT")

    print("Exact-match integration audit")
    for metric, value in con.sql(
        "SELECT metric, value FROM integration_audit WHERE run_id='exact_v2' ORDER BY metric"
    ).fetchall():
        print(f"  {metric}: {int(value):,}")

    con.execute("DETACH spotify")
    con.execute("DETACH listening")
    con.close()


if __name__ == "__main__":
    main()
