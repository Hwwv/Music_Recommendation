"""Run the Content Recommenders"""
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
from music_recommender.models import ContentRecommender

INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
CBM_OUTPUT_DIR = ROOT / "artifacts" / "cbm_test"

DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"
CBM_OUTPUT_VERSION = "cbm_eval20_test_v1"

N_CONTINUOUS = 10
ALLOW_TEST = False 
K = [20]
ALPHA = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Content Recommender")
    parser.add_argument("--dataset_version", type=str, default=DATASET_VERSION, help="Dataset version to use")
    parser.add_argument("--split_version", type=str, default=SPLIT_VERSION, help="Split version to use")
    parser.add_argument("--feature_schema_version", type=str, default=FEATURE_SCHEMA_VERSION, help="Feature schema version to use")
    parser.add_argument("--data_db_path", type=Path, default=INTEGRATION, help="Path to the DuckDB database file")
    parser.add_argument("--cbm_output_version", type=str, default=CBM_OUTPUT_VERSION, help="Output version for CBM results")
    parser.add_argument("--cbm_output_dir", type=str, default=CBM_OUTPUT_DIR, help="Output directory for CBM results")
    parser.add_argument("--allow_test", action="store_true", help="Allow loading of test split (default: False)")
    parser.add_argument("--k", type=int, default=K, help="Final k value for evaluation metrics")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Final alpha value for the Content-Based Recommender")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running Content Recommender with dataset_version={args.dataset_version}, split_version={args.split_version}, feature_schema_version={args.feature_schema_version}")

    # Load the data
    data_loader = MusicDataLoader(
        data_version = args.dataset_version,
        split_version = args.split_version,
        feature_schema_version = args.feature_schema_version,
        data_db_path = args.data_db_path,
        allow_test = args.allow_test,
    )

    # Train the models with different hyperparameters and get results for the test set
    
    experiment = data_loader.load_experiment("test")
    test_users = sorted(experiment.truth)

    # Train and evaluate the Content-Based Recommender (CBM)
    alpha = args.alpha
    print(f"alpha={alpha}")
    cbm_recommender = ContentRecommender(confidence_alpha=alpha)
    cbm_recommender.fit(experiment.train, experiment.features)

    recommendations = {}
    for user in test_users:
        recommendations[user] = cbm_recommender.recommend(user, k=max(args.k))
    assert_unseen_recommendations(recommendations, experiment.seen)

    metrics = evaluate_topk(
        recommendations,
        experiment.truth,
        set(experiment.catalog),
        k_values=args.k,
    )
    print(metrics)

    output_dir = Path(args.cbm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cbm_output = output_dir / f"{CBM_OUTPUT_VERSION}.json"
    cbm_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CBM results saved to {cbm_output.resolve()}")
    data_loader.close()


if __name__ == "__main__":
    main()
