#!/usr/bin/env python3
"""Tune sparse Item-KNN or implicit ALS on the locked validation split (default data version v1 and running for CF)."""

from __future__ import annotations

import argparse
from collections import defaultdict, namedtuple
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / ".tools", ROOT / "src"):
    if path.exists():
        sys.path.insert(0, str(path))

from music_recommender.cf import ImplicitALS, SparseItemKNN, recommend_users
from music_recommender.evaluation import assert_unseen_recommendations, evaluate_topk

import duckdb

OUTPUT_DIR = ROOT / "artifacts" / "cf"
INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
DatabaseRow = namedtuple("DatabaseRow", "user_id feature_cluster_id playcount_raw")


def load_train_and_validation(split_version: str):
    """Load CF data as Python records, without requiring pandas."""
    connection = duckdb.connect(str(INTEGRATION), read_only=True)
    try:
        metadata = connection.execute(
            "SELECT dataset_version FROM feature_split_datasets WHERE split_version = ?",
            [split_version],
        ).fetchone()
        if metadata is None:
            raise ValueError(f"unknown split version: {split_version}")
        dataset_version = str(metadata[0])
        train = [
            DatabaseRow(int(user), str(item), float(playcount))
            for user, item, playcount in connection.execute(
                """
                SELECT s.user_id, s.feature_cluster_id, g.playcount_raw
                FROM feature_dataset_splits s
                JOIN feature_graph_interactions g
                  ON s.user_id = g.user_id
                 AND s.feature_cluster_id = g.feature_cluster_id
                 AND g.dataset_version = ?
                WHERE s.split_version = ? AND s.split = 'train'
                ORDER BY s.user_id, s.feature_cluster_id
                """,
                [dataset_version, split_version],
            ).fetchall()
        ]
        truth: dict[int, set[str]] = defaultdict(set)
        for user, item in connection.execute(
            """
            SELECT user_id, feature_cluster_id
            FROM feature_dataset_splits
            WHERE split_version = ? AND split = 'validation'
            ORDER BY user_id, feature_cluster_id
            """,
            [split_version],
        ).fetchall():
            truth[int(user)].add(str(item))
        return train, dict(truth), dataset_version
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("item-knn", "als"), required=True)
    parser.add_argument("--split-version", default="feature_split_u5_i2_eval20_seed42_v1")
    parser.add_argument("--feature-schema-version", default="feature_matrix_audio_v1")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 10.0, 40.0])
    parser.add_argument("--neighbours", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--weighting", nargs="+", choices=("cosine", "bm25"), default=["cosine", "bm25"])
    parser.add_argument("--min-cooccurrence", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--factors", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--iterations", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-version", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value < 0 for value in args.alpha) or any(value <= 0 for value in args.k):
        raise SystemExit("alpha must be non-negative and k must be positive")
    train, truth, dataset_version = load_train_and_validation(args.split_version)
    users = sorted(truth)
    max_k = max(args.k)
    if args.model == "item-knn":
        configurations = (
            {"alpha": alpha, "neighbours": neighbours, "weighting": weighting,
             "min_cooccurrence": minimum}
            for alpha, neighbours, weighting, minimum in itertools.product(
                args.alpha, args.neighbours, args.weighting, args.min_cooccurrence
            )
        )
    else:
        configurations = (
            {"alpha": alpha, "factors": factors, "regularization": regularization,
             "iterations": iterations, "seed": args.seed}
            for alpha, factors, regularization, iterations in itertools.product(
                args.alpha, args.factors, args.regularization, args.iterations
            )
        )

    runs: list[dict[str, object]] = []
    for number, config in enumerate(configurations, start=1):
        print(f"run {number}: {config}", flush=True)
        started = time.perf_counter()
        model = (SparseItemKNN(**config) if args.model == "item-knn" else ImplicitALS(**config)).fit(train)
        recommendations = recommend_users(model, users, max_k)
        seen = {
            user: {str(model.data.item_ids[i]) for i in model.data.confidence.getrow(
                model.data.user_to_index[user]
            ).indices}
            for user in users if user in model.data.user_to_index
        }
        assert_unseen_recommendations(recommendations, seen)
        metrics = evaluate_topk(recommendations, truth, set(map(str, model.data.item_ids)), args.k)
        metrics["runtime_seconds"] = time.perf_counter() - started
        runs.append({"configuration": config, "metrics": metrics})
        print(json.dumps(metrics, sort_keys=True), flush=True)

    primary = f"ndcg@{max(args.k)}"
    runs.sort(key=lambda run: (-float(run["metrics"][primary]), float(run["metrics"]["runtime_seconds"])))
    output_version = args.output_version or f"{args.model}_validation_v1"
    result = {
        "output_version": output_version,
        "model": args.model,
        "split_version": args.split_version,
        "dataset_version": dataset_version,
        "feature_schema_version": args.feature_schema_version,
        "evaluation_split": "validation",
        "selection_metric": primary,
        "best_configuration": runs[0]["configuration"] if runs else None,
        "runs": runs,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{output_version}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
