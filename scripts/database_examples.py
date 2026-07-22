#!/usr/bin/env python3
"""Read-only examples for the three generated project databases."""

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

DATABASES = ROOT / "data" / "databases"


def print_rows(title: str, columns: list[str], rows: list[tuple]) -> None:
    print(f"\n{title}")
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join("NULL" if value is None else str(value) for value in row))


def spotify_example() -> None:
    con = duckdb.connect(str(DATABASES / "spotify.duckdb"), read_only=True)
    result = con.sql(
        """
        SELECT
            t.track_id, t.track_name, t.artists, t.popularity,
            round(t.energy, 3) AS energy,
            round(t.danceability, 3) AS danceability,
            list(g.track_genre ORDER BY g.track_genre) AS genres
        FROM spotify_tracks AS t
        LEFT JOIN spotify_track_genres AS g USING (track_id)
        WHERE t.popularity >= 80 AND t.energy >= 0.80
        GROUP BY ALL
        ORDER BY t.popularity DESC, t.track_name
        LIMIT 5
        """
    )
    print_rows("SPOTIFY EXAMPLE", [d[0] for d in result.description], result.fetchall())
    con.close()


def listening_example() -> None:
    con = duckdb.connect(str(DATABASES / "listening_clean.duckdb"), read_only=True)
    result = con.sql(
        """
        SELECT
            user_id, source_rank, track_name, artist_name, playcount,
            round(1 + ln(1 + playcount), 3) AS example_confidence
        FROM user_top_tracks
        WHERE user_id = 1
        ORDER BY source_rank
        LIMIT 10
        """
    )
    print_rows("LISTENING EXAMPLE", [d[0] for d in result.description], result.fetchall())
    con.close()


def integration_example() -> None:
    con = duckdb.connect(str(DATABASES / "integration.duckdb"), read_only=True)
    result = con.sql(
        """
        SELECT
            (SELECT count(*) FROM project_users) AS available_users,
            (SELECT count(*) FROM project_tracks) AS available_spotify_tracks,
            (SELECT count(*) FROM track_crosswalk) AS reviewed_crosswalk_rows,
            (SELECT count(*) FROM interactions_integrated) AS integrated_interactions
        """
    )
    print_rows("INTEGRATION EXAMPLE", [d[0] for d in result.description], result.fetchall())
    con.close()


def main() -> None:
    spotify_example()
    listening_example()
    integration_example()


if __name__ == "__main__":
    main()
