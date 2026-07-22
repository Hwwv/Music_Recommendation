#!/usr/bin/env python3
"""Export deterministic exact-match samples for human quality review."""

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
OUT = ROOT / "artifacts" / "matching"
OUT.mkdir(parents=True, exist_ok=True)


def q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main() -> None:
    con = duckdb.connect(str(DB / "integration.duckdb"), read_only=True)
    con.execute(f"ATTACH '{q(DB / 'listening_clean.duckdb')}' AS listening (READ_ONLY)")
    con.execute(f"ATTACH '{q(DB / 'spotify.duckdb')}' AS spotify (READ_ONLY)")

    con.execute(
        f"""
        COPY (
            SELECT
                l.representative_track_name AS listening_track,
                l.representative_artist_name AS listening_artist,
                p.track_name AS spotify_track,
                p.artists AS spotify_artists,
                p.album_name AS spotify_album,
                (
                    SELECT string_agg(g.track_genre, '; ' ORDER BY g.track_genre)
                    FROM spotify.spotify_track_genres g
                    WHERE g.track_id = p.track_id
                ) AS spotify_genres,
                x.spotify_track_id,
                p.duration_ms,
                l.user_count,
                l.interaction_count,
                x.match_tier,
                '' AS review_label,
                '' AS review_reason,
                '' AS reviewer
            FROM track_crosswalk x
            JOIN listening.listening_track_keys l
              USING (track_name_norm, artist_name_norm)
            JOIN spotify.spotify_tracks p ON x.spotify_track_id = p.track_id
            ORDER BY hash(x.track_name_norm, x.artist_name_norm)
            LIMIT 200
        ) TO '{q(OUT / 'accepted_exact_v1_review.csv')}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
            WITH sampled_keys AS (
                SELECT track_name_norm, artist_name_norm
                FROM track_match_decisions
                WHERE decision = 'ambiguous'
                ORDER BY hash(track_name_norm, artist_name_norm)
                LIMIT 100
            )
            SELECT
                l.representative_track_name AS listening_track,
                l.representative_artist_name AS listening_artist,
                c.spotify_track_id,
                p.track_name AS spotify_track,
                p.artists AS spotify_artists,
                p.album_name AS spotify_album,
                (
                    SELECT string_agg(g.track_genre, '; ' ORDER BY g.track_genre)
                    FROM spotify.spotify_track_genres g
                    WHERE g.track_id = p.track_id
                ) AS spotify_genres,
                p.duration_ms,
                p.popularity,
                d.candidate_count,
                c.match_tier,
                '' AS review_action,
                '' AS selected_spotify_track_id,
                '' AS review_reason,
                '' AS reviewer
            FROM sampled_keys k
            JOIN track_match_decisions d
              USING (track_name_norm, artist_name_norm)
            JOIN track_match_candidates c
              USING (track_name_norm, artist_name_norm)
            JOIN listening.listening_track_keys l
              USING (track_name_norm, artist_name_norm)
            JOIN spotify.spotify_tracks p ON c.spotify_track_id = p.track_id
            ORDER BY l.representative_track_name, l.representative_artist_name,
                     p.popularity DESC, c.spotify_track_id
        ) TO '{q(OUT / 'ambiguous_exact_v1_review.csv')}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute("DETACH listening")
    con.execute("DETACH spotify")
    con.close()
    print(OUT / "accepted_exact_v1_review.csv")
    print(OUT / "ambiguous_exact_v1_review.csv")


if __name__ == "__main__":
    main()
