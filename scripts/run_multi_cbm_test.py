"""Run the Multi-Interest Content Recommender"""
from __future__ import annotations

import argparse
import json

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from music_recommender.data_loader import MusicDataLoader
from music_recommender.evaluation import evaluate_topk, assert_unseen_recommendations
from music_recommender.models import MultiInterestContentRecommender

INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
MULTICBM_OUTPUT_DIR = ROOT / "artifacts" / "multicbm_test"

DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"
MULTICBM_OUTPUT_VERSION = "multicbm_eval20_test_v1"

ALLOW_TEST = False 
K = [20]
ALPHA = 0.9
GLOBAL_WEIGHT = 0.3
K_FOR_KMEANS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Multi-Interest Recommenders")
    parser.add_argument("--dataset_version", type=str, default=DATASET_VERSION, help="Dataset version to use")
    parser.add_argument("--split_version", type=str, default=SPLIT_VERSION, help="Split version to use")
    parser.add_argument("--feature_schema_version", type=str, default=FEATURE_SCHEMA_VERSION, help="Feature schema version to use")
    parser.add_argument("--data_db_path", type=Path, default=INTEGRATION, help="Path to the DuckDB database file")
    parser.add_argument("--multicbm_output_version", type=str, default=MULTICBM_OUTPUT_VERSION, help="Output version for Multi-CBM results")
    parser.add_argument("--multicbm_output_dir", type=str, default=MULTICBM_OUTPUT_DIR, help="Output directory for Multi-CBM results")
    parser.add_argument("--allow_test", action="store_true", help="Allow loading of test split (default: False)")
    parser.add_argument("--k", type=int, nargs="+", default=K, help="Final k value for evaluation metrics")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Final alpha value for the Content-Based Recommender")
    parser.add_argument("--global_weight", type=float, default=GLOBAL_WEIGHT, help="Final global weight values for the Multi-Interest Content-Based Recommender")
    parser.add_argument("--k_for_kmeans", type=int, default=K_FOR_KMEANS, help="Final k value for KMeans clustering in the Multi-Interest Content-Based Recommender")
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
    
    train_interactions = data_loader.load_split_interactions("train")
    features = data_loader.load_feature_mappings()
    test_truths = data_loader.load_split_truth("test")
    test_users = sorted(set(test_truths.keys()))

    # Train and evaluate the Multi-Interest Content-Based Recommender (Multi-CBM)
    output_dir = Path(args.multicbm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multicbm_output = output_dir / f"{args.multicbm_output_version}.json"
    alpha, global_weight, km = args.alpha, args.global_weight, args.k_for_kmeans
    print(f"alpha: {alpha}, global weight:{global_weight}, km: {km}")
    multicbm_recommender = MultiInterestContentRecommender(confidence_alpha=alpha, global_weight=global_weight)
    multicbm_recommender.fit(train_interactions, features, k=km)
    seen = multicbm_recommender.seen.copy()
    catalog = multicbm_recommender.catalog

    recommendations = {}
    for user in test_users:
        recommendations[user] = multicbm_recommender.recommend(user, k=max(args.k))
    assert_unseen_recommendations(recommendations, seen)
    metrics = evaluate_topk(recommendations, test_truths, catalog, k_values=args.k)

    serializable_results = {}
    key_str = f"alpha_{alpha}_gw_{global_weight}_km_{km}"
    serializable_results[key_str] = metrics

    multicbm_output.write_text(json.dumps(serializable_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"metrics: {metrics}")

    print(f"Multi-CBM results saved to {multicbm_output.resolve()}")

if __name__ == "__main__":
    main()