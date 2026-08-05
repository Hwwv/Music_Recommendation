#!/usr/bin/env python3
"""Create and audit a deterministic train/validation/test feature-graph split."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="feature_graph_u5_i2_v2")
    parser.add_argument(
        "--split-version", default="feature_split_u5_i2_eval20_seed42_v2"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-evaluation-items", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    return parser.parse_args()


def ensure_registry(con: duckdb.DuckDBPyConnection) -> None:
    """Migrate an existing integration DB without rebuilding it."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_split_datasets (
            split_version VARCHAR PRIMARY KEY,
            dataset_version VARCHAR NOT NULL REFERENCES feature_graph_datasets(dataset_version),
            seed INTEGER NOT NULL,
            min_evaluation_items INTEGER NOT NULL CHECK (min_evaluation_items >= 3),
            validation_fraction DOUBLE NOT NULL CHECK (validation_fraction > 0),
            test_fraction DOUBLE NOT NULL CHECK (test_fraction > 0),
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )


def print_audit(
    con: duckdb.DuckDBPyConnection, split_version: str, reused: bool = False
) -> None:
    state = "already exists" if reused else "created"
    print(f"Feature split {state}: {split_version}")
    for metric, value in con.execute(
        """
        SELECT metric, value FROM integration_audit
        WHERE run_id = ? ORDER BY metric
        """,
        [split_version],
    ).fetchall():
        print(f"  {metric}: {int(value):,}")


def main() -> None:
    args = parse_args()
    if not args.dataset_version.strip() or not args.split_version.strip():
        raise SystemExit("dataset and split versions cannot be empty")
    if args.min_evaluation_items < 3:
        raise SystemExit("--min-evaluation-items must be at least 3")
    if args.validation_fraction <= 0 or args.test_fraction <= 0:
        raise SystemExit("validation and test fractions must be positive")
    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("validation and test fractions must sum to less than 1")

    con = duckdb.connect(str(INTEGRATION))
    con.execute("PRAGMA threads=4")
    ensure_registry(con)
    if not con.execute(
        "SELECT count(*) FROM feature_graph_datasets WHERE dataset_version = ?",
        [args.dataset_version],
    ).fetchone()[0]:
        raise SystemExit(f"unknown dataset version: {args.dataset_version}")

    requested = (
        args.dataset_version,
        args.seed,
        args.min_evaluation_items,
        args.validation_fraction,
        args.test_fraction,
    )
    existing = con.execute(
        """
        SELECT dataset_version, seed, min_evaluation_items,
               validation_fraction, test_fraction
        FROM feature_split_datasets WHERE split_version = ?
        """,
        [args.split_version],
    ).fetchone()
    if existing is not None:
        if existing != requested:
            raise SystemExit(
                f"split version {args.split_version!r} has configuration {existing}; "
                f"choose a new version for {requested}"
            )
        print_audit(con, args.split_version, reused=True)
        con.close()
        return

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE staged_feature_split AS
            WITH ranked AS (
                SELECT
                    user_id,
                    feature_cluster_id,
                    count(*) OVER (PARTITION BY user_id) AS item_count,
                    row_number() OVER (
                        PARTITION BY user_id
                        ORDER BY sha256(concat(
                            ?::VARCHAR, ':', user_id::VARCHAR, ':',
                            feature_cluster_id
                        )), feature_cluster_id
                    ) AS random_order
                FROM feature_graph_interactions
                WHERE dataset_version = ?
            ), allocated AS (
                SELECT
                    *,
                    CASE WHEN item_count >= ?
                        THEN greatest(1, floor(item_count * ?)::INTEGER)
                        ELSE 0 END AS validation_count,
                    CASE WHEN item_count >= ?
                        THEN greatest(1, floor(item_count * ?)::INTEGER)
                        ELSE 0 END AS test_count
                FROM ranked
            )
            SELECT
                user_id,
                feature_cluster_id,
                CASE
                    WHEN random_order <= validation_count THEN 'validation'
                    WHEN random_order <= validation_count + test_count THEN 'test'
                    ELSE 'train'
                END AS split
            FROM allocated
            """,
            [
                args.seed,
                args.dataset_version,
                args.min_evaluation_items,
                args.validation_fraction,
                args.min_evaluation_items,
                args.test_fraction,
            ],
        )
        con.execute(
            """
            INSERT INTO feature_split_datasets (
                split_version, dataset_version, seed, min_evaluation_items,
                validation_fraction, test_fraction
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                args.split_version,
                args.dataset_version,
                args.seed,
                args.min_evaluation_items,
                args.validation_fraction,
                args.test_fraction,
            ],
        )
        con.execute(
            """
            INSERT INTO feature_dataset_splits
            SELECT ?, ?, user_id, feature_cluster_id, split
            FROM staged_feature_split
            """,
            [args.split_version, args.seed],
        )

        checks = con.execute(
            """
            WITH graph AS (
                SELECT user_id, feature_cluster_id
                FROM feature_graph_interactions WHERE dataset_version = ?
            ), assignments AS (
                SELECT user_id, feature_cluster_id, split
                FROM feature_dataset_splits WHERE split_version = ?
            ), user_counts AS (
                SELECT
                    user_id,
                    count(*) AS total_count,
                    count(*) FILTER (WHERE split = 'train') AS train_count,
                    count(*) FILTER (WHERE split = 'validation') AS validation_count,
                    count(*) FILTER (WHERE split = 'test') AS test_count
                FROM assignments GROUP BY user_id
            )
            SELECT
                (SELECT count(*) FROM graph),
                (SELECT count(*) FROM assignments),
                (SELECT count(*) FROM graph g ANTI JOIN assignments a
                    USING (user_id, feature_cluster_id)),
                (SELECT count(*) FROM assignments a ANTI JOIN graph g
                    USING (user_id, feature_cluster_id)),
                (SELECT count(*) FROM user_counts
                    WHERE total_count >= ? AND
                    (train_count = 0 OR validation_count = 0 OR test_count = 0)),
                (SELECT count(*) FROM user_counts
                    WHERE total_count < ? AND
                    (validation_count > 0 OR test_count > 0))
            """,
            [
                args.dataset_version,
                args.split_version,
                args.min_evaluation_items,
                args.min_evaluation_items,
            ],
        ).fetchone()
        source_rows, split_rows, missing, extra, incomplete_eval, sparse_heldout = checks
        if source_rows != split_rows or any(
            (missing, extra, incomplete_eval, sparse_heldout)
        ):
            raise RuntimeError(f"split integrity checks failed: {checks}")

        params: list[object] = []
        audit_parts: list[str] = []

        def add(metric: str, query: str, *query_params: object) -> None:
            audit_parts.append(f"SELECT ?, '{metric}', ({query})")
            params.append(args.split_version)
            params.extend(query_params)

        add(
            "source_interactions",
            "SELECT count(*) FROM feature_graph_interactions WHERE dataset_version = ?",
            args.dataset_version,
        )
        for split in ("train", "validation", "test"):
            add(
                f"{split}_interactions",
                "SELECT count(*) FROM feature_dataset_splits "
                "WHERE split_version = ? AND split = ?",
                args.split_version,
                split,
            )
        add(
            "split_interactions",
            "SELECT count(*) FROM feature_dataset_splits WHERE split_version = ?",
            args.split_version,
        )
        add(
            "evaluation_users",
            "SELECT count(DISTINCT user_id) FROM feature_dataset_splits "
            "WHERE split_version = ? AND split = 'validation'",
            args.split_version,
        )
        add(
            "train_users",
            "SELECT count(DISTINCT user_id) FROM feature_dataset_splits "
            "WHERE split_version = ? AND split = 'train'",
            args.split_version,
        )
        add(
            "train_feature_clusters",
            "SELECT count(DISTINCT feature_cluster_id) FROM feature_dataset_splits "
            "WHERE split_version = ? AND split = 'train'",
            args.split_version,
        )
        cold_filter = """
            SELECT count(*) FROM feature_dataset_splits h
            WHERE h.split_version = ? AND h.split IN ('validation', 'test')
              AND NOT EXISTS (
                SELECT 1 FROM feature_dataset_splits t
                WHERE t.split_version = h.split_version AND t.split = 'train'
                  AND t.feature_cluster_id = h.feature_cluster_id
              )
        """
        add(
            "heldout_interactions_with_item_absent_from_train",
            cold_filter,
            args.split_version,
        )
        add(
            "duplicate_user_item_assignments",
            "SELECT count(*) - count(DISTINCT (user_id, feature_cluster_id)) "
            "FROM feature_dataset_splits WHERE split_version = ?",
            args.split_version,
        )
        con.execute(
            "INSERT INTO integration_audit (run_id, metric, value) "
            + " UNION ALL ".join(audit_parts),
            params,
        )
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    print_audit(con, args.split_version)
    con.close()


if __name__ == "__main__":
    main()
