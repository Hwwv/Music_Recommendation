#!/usr/bin/env python3
"""Run global, user-history-popularity, and random baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
SRC = ROOT / "src"
for path in (TOOLS, SRC):
    if path.exists():
        sys.path.insert(0, str(path))

import duckdb

from music_recommender.baselines import (
    recommend_from_ranking,
    recommend_history_popularity,
    recommend_random,
)
from music_recommender.evaluation import assert_unseen_recommendations, evaluate_topk

INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
OUTPUT_DIR = ROOT / "artifacts" / "baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-version", default="feature_split_u5_i2_eval20_seed42_v1"
    )
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-version", default="baselines_eval20_validation_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_split == "test" and not args.allow_test:
        raise SystemExit("test is locked; pass --allow-test only after model selection is frozen")
    if any(k <= 0 for k in args.k):
        raise SystemExit("all --k values must be positive")
    max_k = max(args.k)
    con = duckdb.connect(str(INTEGRATION), read_only=True)
    split_exists = con.execute(
        "SELECT count(*) FROM feature_split_datasets WHERE split_version = ?",
        [args.split_version],
    ).fetchone()[0]
    if not split_exists:
        raise SystemExit(f"unknown split version: {args.split_version}")

    truth: dict[int, set[str]] = defaultdict(set)
    for user, item in con.execute(
        """
        SELECT user_id, feature_cluster_id FROM feature_dataset_splits
        WHERE split_version = ? AND split = ? ORDER BY user_id
        """,
        [args.split_version, args.evaluation_split],
    ).fetchall():
        truth[user].add(item)
    users = sorted(truth)
    if not users:
        raise RuntimeError("evaluation split has no users")

    seen: dict[int, set[str]] = defaultdict(set)
    cursor = con.execute(
        """
        SELECT s.user_id, s.feature_cluster_id
        FROM feature_dataset_splits s
        JOIN (SELECT DISTINCT user_id FROM feature_dataset_splits
              WHERE split_version = ? AND split = ?) e USING (user_id)
        WHERE s.split_version = ? AND s.split = 'train'
        ORDER BY s.user_id
        """,
        [args.split_version, args.evaluation_split, args.split_version],
    )
    while rows := cursor.fetchmany(100_000):
        for user, item in rows:
            seen[user].add(item)

    item_rows = con.execute(
        """
        WITH train_stats AS (
            SELECT
                s.feature_cluster_id,
                count(DISTINCT s.user_id) AS listener_count,
                sum(i.confidence_log) AS confidence_sum
            FROM feature_dataset_splits s
            JOIN feature_graph_interactions i
              ON s.user_id = i.user_id
             AND s.feature_cluster_id = i.feature_cluster_id
            JOIN feature_split_datasets d
              ON s.split_version = d.split_version
             AND i.dataset_version = d.dataset_version
            WHERE s.split_version = ? AND s.split = 'train'
            GROUP BY s.feature_cluster_id
        )
        SELECT
            x.feature_cluster_id,
            x.listener_count,
            x.confidence_sum,
            coalesce(c.canonical_popularity, 0)::INTEGER AS spotify_popularity
        FROM train_stats x
        JOIN spotify_feature_clusters c
          ON x.feature_cluster_id = c.feature_cluster_id
        ORDER BY x.feature_cluster_id
        """,
        [args.split_version],
    ).fetchall()
    catalog = [row[0] for row in item_rows]
    catalog_set = set(catalog)
    global_ranking = [
        row[0] for row in sorted(item_rows, key=lambda row: (-row[1], -row[2], row[0]))
    ]

    target_rows = con.execute(
        """
        SELECT
            s.user_id,
            round(sum(coalesce(c.canonical_popularity, 0) * i.confidence_log)
                  / sum(i.confidence_log))::INTEGER AS target_popularity
        FROM feature_dataset_splits s
        JOIN feature_graph_interactions i
          ON s.user_id = i.user_id AND s.feature_cluster_id = i.feature_cluster_id
        JOIN feature_split_datasets d
          ON s.split_version = d.split_version AND i.dataset_version = d.dataset_version
        JOIN spotify_feature_clusters c
          ON s.feature_cluster_id = c.feature_cluster_id
        JOIN (SELECT DISTINCT user_id FROM feature_dataset_splits
              WHERE split_version = ? AND split = ?) e
          ON s.user_id = e.user_id
        WHERE s.split_version = ? AND s.split = 'train'
        GROUP BY s.user_id
        """,
        [args.split_version, args.evaluation_split, args.split_version],
    ).fetchall()
    targets = {user: max(0, min(100, target)) for user, target in target_rows}
    used_targets = sorted(set(targets.values()))
    history_rankings = {
        target: [
            row[0]
            for row in sorted(
                item_rows,
                key=lambda row: (abs(row[3] - target), -row[1], -row[2], row[0]),
            )
        ]
        for target in used_targets
    }
    con.close()

    generators = {
        "global_popularity": lambda: recommend_from_ranking(
            users, seen, global_ranking, max_k
        ),
        "user_history_popularity": lambda: recommend_history_popularity(
            users, seen, targets, history_rankings, max_k
        ),
        "random": lambda: recommend_random(users, seen, catalog, max_k, args.seed),
    }
    results: dict[str, object] = {
        "output_version": args.output_version,
        "split_version": args.split_version,
        "evaluation_split": args.evaluation_split,
        "seed": args.seed,
        "k_values": sorted(set(args.k)),
        "catalog_items": len(catalog),
        "models": {},
    }
    for name, generate in generators.items():
        started = time.perf_counter()
        recommendations = generate()
        assert_unseen_recommendations(recommendations, seen)
        metrics = evaluate_topk(recommendations, truth, catalog_set, args.k)
        metrics["runtime_seconds"] = time.perf_counter() - started
        results["models"][name] = metrics
        print(name)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.6f}")
            else:
                print(f"  {metric}: {value:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{args.output_version}.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
