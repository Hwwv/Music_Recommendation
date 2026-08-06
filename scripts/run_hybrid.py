#!/usr/bin/env python3
"""Tune sparse Item-KNN or implicit ALS on the locked validation split."""

from __future__ import annotations

import argparse
from collections import defaultdict, namedtuple
import itertools
import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / ".tools", ROOT / "src"):
    if path.exists():
        sys.path.insert(0, str(path))

from music_recommender.data_loader import MusicDataLoader
from music_recommender.cf import SparseItemKNN
from music_recommender.evaluation import assert_unseen_recommendations, evaluate_topk
from music_recommender.models import ContentRecommender

import duckdb

OUTPUT_DIR = ROOT / "artifacts" / "hybrid"
INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"
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
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--split-version", default=SPLIT_VERSION)
    parser.add_argument("--feature-schema-version", default=FEATURE_SCHEMA_VERSION)
    parser.add_argument("--data_db_path", type=Path, default=INTEGRATION, help="Path to the DuckDB database file")
    parser.add_argument("--allow_test", action="store_true", help="Allow loading of test split (default: False)")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0])
    parser.add_argument("--neighbours", type=int, nargs="+", default=[200])
    parser.add_argument("--weighting", nargs="+", choices=("cosine", "bm25"), default=["bm25"])
    parser.add_argument("--min-cooccurrence", type=int, nargs="+", default=[2])
    parser.add_argument("--k", type=int, nargs="+", default=[20])
    parser.add_argument("--cf_weight", nargs="+", default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    parser.add_argument("--output-version", default=None)
    return parser.parse_args()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    denom = row_max - row_min
    return np.where(denom > 0, (matrix-row_min) / denom, 1.0)


def main() -> None:
    args = parse_args()
    if any(value < 0 for value in args.alpha) or any(value <= 0 for value in args.k):
        raise SystemExit("alpha must be non-negative and k must be positive")
    train, truth, dataset_version = load_train_and_validation(args.split_version)
    users = sorted(truth)
    max_k = max(args.k)
    configurations = (
        {"alpha": alpha, "neighbours": neighbours, "weighting": weighting,
            "min_cooccurrence": minimum}
        for alpha, neighbours, weighting, minimum in itertools.product(
            args.alpha, args.neighbours, args.weighting, args.min_cooccurrence
        )
    )

    data_loader = MusicDataLoader(
        data_version = args.dataset_version,
        split_version = args.split_version,
        feature_schema_version = args.feature_schema_version,
        data_db_path = args.data_db_path,
        allow_test = args.allow_test,
    )
    train_interactions = data_loader.load_split_interactions("train")
    features = data_loader.load_feature_mappings()
    features = {item: vec for item, vec in features.items()}

    runs: list[dict[str, object]] = []
    for number, config in enumerate(configurations, start=1):
        print(f"run {number}: {config}", flush=True)
        cf_model = (SparseItemKNN(**config)).fit(train)
        cbm_recommender = ContentRecommender(confidence_alpha=config["alpha"])
        cbm_recommender.fit(train_interactions, features)

        cf_weights = args.cf_weight

        common_items = sorted(set(cf_model.data.item_ids) & set(cbm_recommender.catalog))
        item_index = {item: i for i, item in enumerate(common_items)}
        n_items = len(common_items)

        cf_item_pos = np.array([cf_model.data.item_to_index[item] for item in common_items])

        cf_matrix = np.zeros((len(users), n_items), dtype=np.float32)
        cbm_matrix = np.zeros((len(users), n_items), dtype=np.float32)
        seen_mask = np.zeros((len(users), n_items), dtype=bool)

        seen_cf = {
            user: {str(cf_model.data.item_ids[i]) for i in cf_model.data.confidence.getrow(
                cf_model.data.user_to_index[user]
                ).indices}
                for user in users if user in cf_model.data.user_to_index
                }

        for row, user in enumerate(users):
            full_cf_scores = cf_model.score_not_processed(user) 
            cf_matrix[row] = full_cf_scores[cf_item_pos]

            cbm_scores = cbm_recommender.score(user)  # dict
            cbm_matrix[row] = [cbm_scores.get(item, 0.0) for item in common_items]

            seen = seen_cf.get(user, set()) | cbm_recommender.seen.get(user, set())
            seen_mask[row] = [item in seen for item in common_items]

        cf_matrix = normalize_rows(cf_matrix)
        cbm_matrix = normalize_rows(cbm_matrix)

        for cf_weight in cf_weights:
            combined = cf_weight * cf_matrix + (1 - cf_weight) * cbm_matrix
            combined[seen_mask] = -np.inf 

            recommendations = {}
            for row, user in enumerate(users):
                top_idx = np.argpartition(combined[row], -max_k)[-max_k:]
                top_idx = top_idx[np.argsort(-combined[row, top_idx])]
                recommendations[user] = [common_items[i] for i in top_idx if np.isfinite(combined[row, i])]

            assert_unseen_recommendations(recommendations, seen_cf)
            assert_unseen_recommendations(recommendations, dict(cbm_recommender.seen))
            
            metrics = evaluate_topk(recommendations, truth, set(common_items), args.k)
            runs.append({"configuration": config, "cf_weight": cf_weight, "metrics": metrics})
            print(json.dumps(metrics, sort_keys=True), flush=True)

    primary = f"ndcg@{max(args.k)}"
    runs.sort(key=lambda run: (-float(run["metrics"][primary])))
    output_version = args.output_version or f"hybrid_validation_v1"
    result = {
        "output_version": output_version,
        "split_version": args.split_version,
        "dataset_version": dataset_version,
        "feature_schema_version": args.feature_schema_version,
        "evaluation_split": "validation",
        "selection_metric": primary,
        "best_cf_weight": runs[0]["cf_weight"] if runs else None,
        "runs": runs,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{output_version}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
