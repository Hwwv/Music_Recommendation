#!/usr/bin/env python3
"""Run global, user-history-popularity, and random baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
SRC = ROOT / "src"
for path in (TOOLS, SRC):
    if path.exists():
        sys.path.insert(0, str(path))

import numpy as np

from music_recommender.baselines import (
    recommend_from_ranking,
    recommend_history_popularity,
    recommend_random,
)
from music_recommender.evaluation import assert_unseen_recommendations, evaluate_topk
from music_recommender.data_loader import MusicDataLoader

INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
OUTPUT_DIR = ROOT / "artifacts" / "baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-version", default="feature_split_u5_i2_eval20_seed42_v1"
    )
    parser.add_argument("--feature-schema-version", default="feature_matrix_audio_v1")
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
    loader = MusicDataLoader(
        split_version=args.split_version,
        feature_schema_version=args.feature_schema_version,
        data_db_path=INTEGRATION,
        allow_test=args.evaluation_split == "test" and args.allow_test,
    )
    experiment = loader.load_experiment(args.evaluation_split)
    item_ids = list(experiment.catalog)
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    truth = {
        int(user): {item_to_index[str(item)] for item in items if str(item) in item_to_index}
        for user, items in experiment.truth.items()
    }
    truth = {user: items for user, items in truth.items() if items}
    users = sorted(truth)
    if not users:
        raise RuntimeError("evaluation split has no users")

    seen: dict[int, set[int]] = {user: set() for user in users}
    listener_counts = np.zeros(len(item_ids), dtype=np.int64)
    confidence_sums = np.zeros(len(item_ids), dtype=np.float64)
    user_history: dict[int, list[tuple[int, float]]] = {user: [] for user in users}
    for row in experiment.train:
        user = int(row.user_id)
        item = item_to_index.get(str(row.item_id))
        if item is None:
            continue
        confidence = 1.0 + np.log1p(max(0.0, float(row.play_count)))
        listener_counts[item] += 1
        confidence_sums[item] += confidence
        if user in seen:
            seen[user].add(item)
            user_history[user].append((item, confidence))

    catalog = list(range(len(item_ids)))
    catalog_set = set(catalog)
    global_ranking = [
        item for item in sorted(
            catalog,
            key=lambda item: (
                -listener_counts[item], -confidence_sums[item], item_ids[item]
            ),
        )
    ]
    popularity_rows = loader.execute_query(
        """
        SELECT feature_cluster_id, canonical_popularity
        FROM spotify_feature_clusters
        """
    )
    popularity_by_item = {
        str(row.feature_cluster_id): int(row.canonical_popularity)
        for row in popularity_rows.itertuples(index=False)
    }
    item_popularity = np.asarray(
        [popularity_by_item.get(item_id, 0) for item_id in item_ids], dtype=np.int16
    )
    targets: dict[int, int] = {}
    for user in users:
        history = user_history[user]
        if not history:
            targets[user] = 0
            continue
        total_weight = sum(weight for _, weight in history)
        average = sum(weight * item_popularity[item] for item, weight in history) / total_weight
        targets[user] = max(0, min(100, int(np.floor(average + 0.5))))
    used_targets = sorted(set(targets.values()))
    history_rankings = {
        target: [
            item
            for item in sorted(
                catalog,
                key=lambda item: (
                    abs(int(item_popularity[item]) - target),
                    -listener_counts[item],
                    -confidence_sums[item],
                    item_ids[item],
                ),
            )
        ]
        for target in used_targets
    }

    generators = {
        "global_popularity": lambda: recommend_from_ranking(
            users, seen, global_ranking, max_k
        ),
        "user_history_popularity": lambda: recommend_history_popularity(
            users, seen, targets, history_rankings, max_k
        ),
        "random": lambda: recommend_random(
            users,
            seen,
            catalog,
            max_k,
            args.seed,
            {user: user for user in users},
        ),
    }
    results: dict[str, object] = {
        "output_version": args.output_version,
        "dataset_version": experiment.dataset_version,
        "split_version": args.split_version,
        "feature_schema_version": args.feature_schema_version,
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
    loader.close()


if __name__ == "__main__":
    main()
