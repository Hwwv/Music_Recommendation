"""Run the Content and Multi-Interest Recommenders"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import numpy as np

from pathlib import Path
import sys
import time

from src.music_recommender.data_loader import MusicDataLoader
from src.music_recommender.evaluation import evaluate_topk, assert_unseen_recommendations
from src.music_recommender.models import MultiInterestContentRecommender

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
MULTICBM_OUTPUT_DIR = ROOT / "artifacts" / "multicbm"

DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"
MULTICBM_OUTPUT_VERSION = "multicbm_eval20_validation_v1"

ALLOW_TEST = False 
KS = [10, 20]
ALPHAS = np.linspace(0.1, 1.0, 10)
GLOBAL_WEIGHTS = np.linspace(0.1, 1.0, 10)
K_FOR_KMEANS = list(range(2, 21, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Content and Multi-Interest Recommenders")
    parser.add_argument("--dataset_version", type=str, default=DATASET_VERSION, help="Dataset version to use")
    parser.add_argument("--split_version", type=str, default=SPLIT_VERSION, help="Split version to use")
    parser.add_argument("--feature_schema_version", type=str, default=FEATURE_SCHEMA_VERSION, help="Feature schema version to use")
    parser.add_argument("--data_db_path", type=Path, default=INTEGRATION, help="Path to the DuckDB database file")
    parser.add_argument("--multicbm_output_version", type=str, default=MULTICBM_OUTPUT_VERSION, help="Output version for Multi-CBM results")
    parser.add_argument("--multicbm_output_dir", type=str, default=MULTICBM_OUTPUT_DIR, help="Output directory for Multi-CBM results")
    parser.add_argument("--allow_test", action="store_true", help="Allow loading of test split (default: False)")
    parser.add_argument("--ks", type=int, nargs="+", default=KS, help="List of k values for evaluation metrics")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS, help="List of alpha values for the Content-Based Recommender")
    parser.add_argument("--global_weights", type=float, nargs="+", default=GLOBAL_WEIGHTS, help="List of global weight values for the Multi-Interest Content-Based Recommender")
    parser.add_argument("--k_for_kmeans", type=int, nargs="+", default=K_FOR_KMEANS, help="List of k values for KMeans clustering in the Multi-Interest Content-Based Recommender")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running Content and Multi-Interest Recommenders with dataset_version={args.dataset_version}, split_version={args.split_version}, feature_schema_version={args.feature_schema_version}")

    # Load the data
    data_loader = MusicDataLoader(
        data_version = args.dataset_version,
        split_version = args.split_version,
        feature_schema_version = args.feature_schema_version,
        data_db_path = args.data_db_path,
        allow_test = args.allow_test,
    )

    # Train the models with different hyperparameters and get results for the validation set
    results_multicbm: dict[tuple[float, float, int], dict] = defaultdict(dict)

    train_interactions = data_loader.load_split_interactions("train")
    features = data_loader.load_feature_mappings()
    validation_truths = data_loader.load_split_truth("validation")
    validation_users = sorted(set(validation_truths.keys()))

    # Train and evaluate the Multi-Interest Content-Based Recommender (Multi-CBM)
    for alpha in args.alphas:
        for global_weight in args.global_weights:
            for km in args.k_for_kmeans:
                multicbm_recommender = MultiInterestContentRecommender(confidence_alpha=alpha, global_weight=global_weight)
                multicbm_recommender.fit(train_interactions, features, k=km)
                seen = multicbm_recommender.seen.copy()
                catalog = multicbm_recommender.catalog

                recommendations = {}
                for user in validation_users:
                    recommendations[user] = multicbm_recommender.recommend(user, k=max(args.ks))
                assert_unseen_recommendations(recommendations, seen)
                metrics = evaluate_topk(recommendations, validation_truths, catalog, k_values=args.ks)
                results_multicbm[(alpha, global_weight, km)] = metrics

    output_dir = Path(args.multicbm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multicbm_output = output_dir / f"{args.multicbm_output_version}.json"
    multicbm_output.write_text(json.dumps(results_multicbm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Multi-CBM results saved to {multicbm_output.resolve()}")
