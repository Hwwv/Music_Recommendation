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

    con.execute(
        f"""
        COPY (
            SELECT
                l.representative_track_name AS listening_track,
                l.representative_artist_name AS listening_artist,
                p.track_name AS spotify_track,
                p.artists AS spotify_artists,
                x.spotify_track_id,
                l.user_count,
                l.interaction_count,
                x.match_tier
            FROM track_crosswalk x
            JOIN listening.listening_track_keys l
              USING (track_name_norm, artist_name_norm)
            JOIN project_tracks p ON x.spotify_track_id = p.track_id
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
                p.popularity,
                d.candidate_count,
                c.match_tier
            FROM sampled_keys k
            JOIN track_match_decisions d
              USING (track_name_norm, artist_name_norm)
            JOIN track_match_candidates c
              USING (track_name_norm, artist_name_norm)
            JOIN listening.listening_track_keys l
              USING (track_name_norm, artist_name_norm)
            JOIN project_tracks p ON c.spotify_track_id = p.track_id
            ORDER BY l.representative_track_name, l.representative_artist_name,
                     p.popularity DESC, c.spotify_track_id
        ) TO '{q(OUT / 'ambiguous_exact_v1_review.csv')}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute("DETACH listening")
    con.close()
    print(OUT / "accepted_exact_v1_review.csv")
    print(OUT / "ambiguous_exact_v1_review.csv")


if __name__ == "__main__":
    main()
