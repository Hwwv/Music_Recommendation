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
from music_recommender.data_loader import load_experiment_data

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
    data = load_experiment_data(
        INTEGRATION,
        args.split_version,
        args.feature_schema_version,
        allow_test=args.evaluation_split == "test" and args.allow_test,
    )
    truth = (
        data.validation_truth if args.evaluation_split == "validation" else data.test_truth
    )
    if truth is None:
        raise RuntimeError("requested evaluation truth was not loaded")
    users = sorted(truth)
    if not users:
        raise RuntimeError("evaluation split has no users")

    seen = {
        user: set(map(int, data.train_binary[user].indices))
        for user in users
    }
    catalog = list(range(len(data.item_ids)))
    catalog_set = set(catalog)
    listener_counts = np.asarray(data.train_binary.sum(axis=0)).ravel()
    confidence = data.confidence(alpha=1.0)
    confidence_sums = np.asarray(confidence.sum(axis=0)).ravel()
    global_ranking = [
        item for item in sorted(
            catalog,
            key=lambda item: (
                -listener_counts[item], -confidence_sums[item], data.item_ids[item]
            ),
        )
    ]
    targets: dict[int, int] = {}
    for user in users:
        row = confidence.getrow(user)
        average = float(row.data @ data.item_popularity[row.indices]) / float(row.data.sum())
        targets[user] = max(0, min(100, int(np.floor(average + 0.5))))
    used_targets = sorted(set(targets.values()))
    history_rankings = {
        target: [
            item
            for item in sorted(
                catalog,
                key=lambda item: (
                    abs(int(data.item_popularity[item]) - target),
                    -listener_counts[item],
                    -confidence_sums[item],
                    data.item_ids[item],
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
            {user: int(data.user_ids[user]) for user in users},
        ),
    }
    results: dict[str, object] = {
        "output_version": args.output_version,
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


if __name__ == "__main__":
    main()
