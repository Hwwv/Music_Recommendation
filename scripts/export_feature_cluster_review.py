#!/usr/bin/env python3
"""Export deterministic review samples at the feature-cluster level."""

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
                x.feature_cluster_id,
                x.spotify_id_count,
                c.canonical_track_id,
                p.track_name AS canonical_spotify_track,
                p.artists AS canonical_spotify_artists,
                p.album_name AS canonical_album,
                (
                    SELECT string_agg(DISTINCT m.album_name, '; ' ORDER BY m.album_name)
                    FROM spotify_feature_cluster_members m
                    WHERE m.feature_cluster_id = x.feature_cluster_id
                ) AS member_albums,
                c.duration_ms,
                c.canonical_popularity,
                l.user_count,
                l.interaction_count,
                '' AS review_label,
                '' AS review_reason,
                '' AS reviewer
            FROM listening_feature_crosswalk x
            JOIN listening.listening_track_keys l
              USING (track_name_norm, artist_name_norm)
            JOIN spotify_feature_clusters c USING (feature_cluster_id)
            JOIN spotify.spotify_tracks p ON c.canonical_track_id = p.track_id
            WHERE x.spotify_id_count > 1
            ORDER BY hash(x.track_name_norm, x.artist_name_norm)
            LIMIT 200
        ) TO '{q(OUT / 'accepted_feature_equivalent_v1_review.csv')}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        COPY (
            WITH sampled_keys AS (
                SELECT track_name_norm, artist_name_norm
                FROM listening_feature_decisions
                WHERE decision = 'ambiguous'
                ORDER BY hash(track_name_norm, artist_name_norm)
                LIMIT 100
            )
            SELECT
                l.representative_track_name AS listening_track,
                l.representative_artist_name AS listening_artist,
                d.feature_cluster_count,
                d.spotify_candidate_count,
                f.feature_cluster_id,
                f.spotify_id_count AS ids_in_this_cluster,
                c.canonical_track_id,
                p.track_name AS canonical_spotify_track,
                p.artists AS canonical_spotify_artists,
                p.album_name AS canonical_album,
                (
                    SELECT string_agg(DISTINCT m.album_name, '; ' ORDER BY m.album_name)
                    FROM spotify_feature_cluster_members m
                    WHERE m.feature_cluster_id = f.feature_cluster_id
                ) AS member_albums,
                c.duration_ms,
                c.explicit,
                c.danceability,
                c.energy,
                c.loudness,
                c.acousticness,
                c.instrumentalness,
                c.valence,
                c.tempo,
                c.canonical_popularity,
                '' AS review_action,
                '' AS selected_feature_cluster_id,
                '' AS review_reason,
                '' AS reviewer
            FROM sampled_keys k
            JOIN listening_feature_decisions d
              USING (track_name_norm, artist_name_norm)
            JOIN listening_feature_candidates f
              USING (track_name_norm, artist_name_norm)
            JOIN listening.listening_track_keys l
              USING (track_name_norm, artist_name_norm)
            JOIN spotify_feature_clusters c USING (feature_cluster_id)
            JOIN spotify.spotify_tracks p ON c.canonical_track_id = p.track_id
            ORDER BY l.representative_track_name, l.representative_artist_name,
                     c.canonical_popularity DESC, f.feature_cluster_id
        ) TO '{q(OUT / 'ambiguous_feature_clusters_v1_review.csv')}'
        (HEADER, DELIMITER ',')
        """
    )

    con.execute("DETACH listening")
    con.execute("DETACH spotify")
    con.close()
    print(OUT / "accepted_feature_equivalent_v1_review.csv")
    print(OUT / "ambiguous_feature_clusters_v1_review.csv")


if __name__ == "__main__":
    main()
