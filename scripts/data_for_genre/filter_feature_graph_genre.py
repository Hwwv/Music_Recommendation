#!/usr/bin/env python3
"""Build a versioned, iteratively filtered user-feature-cluster graph."""

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
except ImportError as exc:
    raise SystemExit("DuckDB is required; see data/databases/README.md") from exc

INTEGRATION = ROOT / "data" / "databases2" / "integration.duckdb"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively prune sparse users and items from the feature interaction graph."
    )
    parser.add_argument("--min-user-items", type=positive_integer, default=5)
    parser.add_argument("--min-item-users", type=positive_integer, default=2)
    parser.add_argument("--dataset-version", default="feature_graph_u5_i2_v2")
    parser.add_argument("--source_run_id", default="feature_cluster_v2")
    return parser.parse_args()


def ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Migrate an existing integration DB without requiring a full rebuild."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_graph_datasets (
            dataset_version VARCHAR PRIMARY KEY,
            min_user_items INTEGER NOT NULL CHECK (min_user_items > 0),
            min_item_users INTEGER NOT NULL CHECK (min_item_users > 0),
            source_run_id VARCHAR NOT NULL,
            iteration_count INTEGER NOT NULL CHECK (iteration_count >= 0),
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_graph_interactions (
            dataset_version VARCHAR NOT NULL REFERENCES feature_graph_datasets(dataset_version),
            user_id INTEGER NOT NULL REFERENCES project_users(user_id),
            feature_cluster_id VARCHAR NOT NULL REFERENCES spotify_feature_clusters(feature_cluster_id),
            canonical_track_id VARCHAR NOT NULL REFERENCES project_tracks(track_id),
            playcount_raw BIGINT NOT NULL CHECK (playcount_raw > 0),
            preference SMALLINT NOT NULL CHECK (preference = 1),
            confidence_log DOUBLE NOT NULL CHECK (confidence_log >= 1),
            source_rank SMALLINT,
            merged_listening_key_count INTEGER NOT NULL CHECK (merged_listening_key_count > 0),
            PRIMARY KEY (dataset_version, user_id, feature_cluster_id)
        )
        """
    )


def main() -> None:
    args = parse_args()
    if not args.dataset_version.strip():
        raise SystemExit("--dataset-version cannot be empty")

    con = duckdb.connect(str(INTEGRATION))
    con.execute("PRAGMA threads=4")
    ensure_tables(con)
    existing = con.execute(
        """
        SELECT min_user_items, min_item_users
        FROM feature_graph_datasets
        WHERE dataset_version = ?
        """,
        [args.dataset_version],
    ).fetchone()
    if existing is not None:
        requested = (args.min_user_items, args.min_item_users)
        if existing != requested:
            raise SystemExit(
                f"dataset version {args.dataset_version!r} already uses thresholds "
                f"{existing}; choose a new --dataset-version for {requested}"
            )
        print(f"Feature graph dataset already exists: {args.dataset_version}")
        print(
            f"  thresholds: user >= {args.min_user_items} items, "
            f"item >= {args.min_item_users} users"
        )
        for metric, value in con.execute(
            """
            SELECT metric, value
            FROM integration_audit
            WHERE run_id = ?
            ORDER BY metric
            """,
            [args.dataset_version],
        ).fetchall():
            print(f"  {metric}: {int(value):,}")
        con.close()
        return

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE filtered_feature_graph AS
            SELECT * FROM feature_interactions_integrated
            """
        )

        iteration_count = 0
        while True:
            before = con.execute(
                "SELECT count(*) FROM filtered_feature_graph"
            ).fetchone()[0]
            con.execute(
                """
                DELETE FROM filtered_feature_graph
                WHERE user_id IN (
                    SELECT user_id
                    FROM filtered_feature_graph
                    GROUP BY user_id
                    HAVING count(*) < ?
                )
                """,
                [args.min_user_items],
            )
            con.execute(
                """
                DELETE FROM filtered_feature_graph
                WHERE feature_cluster_id IN (
                    SELECT feature_cluster_id
                    FROM filtered_feature_graph
                    GROUP BY feature_cluster_id
                    HAVING count(*) < ?
                )
                """,
                [args.min_item_users],
            )
            after = con.execute(
                "SELECT count(*) FROM filtered_feature_graph"
            ).fetchone()[0]
            if after == before:
                break
            iteration_count += 1

        con.execute(
            """
            INSERT INTO feature_graph_datasets (
                dataset_version, min_user_items, min_item_users,
                source_run_id, iteration_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                args.dataset_version,
                args.min_user_items,
                args.min_item_users,
                args.source_run_id,
                iteration_count,
            ],
        )
        con.execute(
            """
            INSERT INTO feature_graph_interactions
            SELECT
                ?,
                user_id,
                feature_cluster_id,
                canonical_track_id,
                playcount_raw,
                preference,
                confidence_log,
                source_rank,
                merged_listening_key_count
            FROM filtered_feature_graph
            """,
            [args.dataset_version],
        )
        con.execute(
            """
            INSERT INTO integration_audit (run_id, metric, value)
            SELECT ?, 'source_interactions', count(*) FROM feature_interactions_integrated
            UNION ALL SELECT ?, 'retained_interactions', count(*) FROM filtered_feature_graph
            UNION ALL SELECT ?, 'retained_users', count(DISTINCT user_id) FROM filtered_feature_graph
            UNION ALL SELECT ?, 'retained_feature_clusters', count(DISTINCT feature_cluster_id) FROM filtered_feature_graph
            UNION ALL SELECT ?, 'iterations_with_removals', ?
            UNION ALL SELECT ?, 'minimum_user_degree', coalesce(min(degree), 0)
                FROM (SELECT count(*) AS degree FROM filtered_feature_graph GROUP BY user_id)
            UNION ALL SELECT ?, 'minimum_item_degree', coalesce(min(degree), 0)
                FROM (SELECT count(*) AS degree FROM filtered_feature_graph GROUP BY feature_cluster_id)
            """,
            [
                args.dataset_version,
                args.dataset_version,
                args.dataset_version,
                args.dataset_version,
                args.dataset_version,
                iteration_count,
                args.dataset_version,
                args.dataset_version,
            ],
        )

        violations = con.execute(
            """
            SELECT
                (SELECT count(*) FROM (
                    SELECT user_id FROM filtered_feature_graph
                    GROUP BY user_id HAVING count(*) < ?
                )),
                (SELECT count(*) FROM (
                    SELECT feature_cluster_id FROM filtered_feature_graph
                    GROUP BY feature_cluster_id HAVING count(*) < ?
                ))
            """,
            [args.min_user_items, args.min_item_users],
        ).fetchone()
        if violations != (0, 0):
            raise RuntimeError(f"degree constraints failed: {violations}")
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    print(f"Feature graph dataset: {args.dataset_version}")
    print(
        f"  thresholds: user >= {args.min_user_items} items, "
        f"item >= {args.min_item_users} users"
    )
    for metric, value in con.execute(
        """
        SELECT metric, value
        FROM integration_audit
        WHERE run_id = ?
        ORDER BY metric
        """,
        [args.dataset_version],
    ).fetchall():
        print(f"  {metric}: {int(value):,}")
    con.close()


if __name__ == "__main__":
    main()
