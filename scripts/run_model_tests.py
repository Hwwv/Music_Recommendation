"""Assess model performances on the test dataset using configurations decided based on the validation dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict, namedtuple
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / ".tools", ROOT / "src"):
    if path.exists():
        sys.path.insert(0, str(path))

from music_recommender.data_loader import MusicDataLoader
from music_recommender.cf import SparseItemKNN, ImplicitALS, recommend_users
from music_recommender.evaluation import assert_unseen_recommendations, evaluate_topk
from music_recommender.models import ContentRecommender, MultiInterestContentRecommender

import duckdb

CBM_PARAMS = {"alpha": 0.8}
MULTI_CBM_PARAMS = {"alpha": 0.9, "global_weight": 0.3, "k_for_kmeans": 16}
CF_PARAMS = {"alpha": 1.0,
             "min_cooccurrence": 2,
             "neighbours": 200,
             "weighting": "bm25"}
ALS_PARAMS = {
    "alpha": 10.0,
    "factors": 128,
    "iterations": 20,
    "regularization": 0.1
    }
HYBRID_PARAMS = {"cf_weight": 0.5}

BASELINE_DIR = ROOT / "artifacts" / "baselines" / "baselines_eval20_test_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "test2"
INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"
OUTPUT_VERSION = "test_v1"
DatabaseRow = namedtuple("DatabaseRow", "user_id feature_cluster_id playcount_raw")
K = [10, 20]


def load_train_and_test(split_version: str):
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
            WHERE split_version = ? AND split = 'test'
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
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly allow loading the locked test split",
    )
    parser.add_argument("--cbm_params", default=CBM_PARAMS)
    parser.add_argument("--cf_params", default=CF_PARAMS)
    parser.add_argument("--multicbm_params", default=MULTI_CBM_PARAMS)
    parser.add_argument("--als_params", default=ALS_PARAMS)
    parser.add_argument("--hybrid_params", default=HYBRID_PARAMS)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-version", default=OUTPUT_VERSION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    denom = row_max - row_min
    return np.where(denom > 0, (matrix-row_min) / denom, 1.0)


def plot_result_bars(results: dict, metric_names: list, label_config: bool = False, configurations: dict = None, output_dir=None, baseline=None):
    model_names = results.keys()
    data = {}
    if baseline:
        baseline_names = baseline.keys()
        for label in baseline_names:
            data[label] = [baseline[label].get(metric, np.nan) for metric in metric_names]

    for model in model_names:
        if label_config and configurations and model in configurations:
            label = f"{model} ({configurations[model]})"
        else:
            label = model
        data[label] = [results[model].get(metric, np.nan) for metric in metric_names]

    data['metrics'] = list(metric_names)
    data_df = pd.DataFrame(data)
    data_df.set_index('metrics', inplace=True)

    ax = data_df.plot(kind='bar', figsize=(12, 6), width=0.8, colormap='Set2')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Values', fontsize=12, fontweight='bold')
    ax.set_title('Metric values for different models', fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(axis='y', alpha=0.3)

    for container in ax.containers:
        values = [v for v in container.datavalues]
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=6)
    plt.tight_layout()

    if output_dir:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir, dpi=150, bbox_inches='tight')
        print(f"plot saved to {output_dir}")

    plt.show()
    

def main() -> dict:
    args = parse_args()
    if not args.allow_test:
        raise SystemExit(
            "Refusing to access the locked test split without --allow-test."
        )

    # load the dataset and configurations for cf and als
    train, truth, dataset_version = load_train_and_test(args.split_version)
    users = sorted(truth)
    ks = args.k
    max_k = max(ks)
    cf_configurations = (
        args.cf_params
    )
    als_configurations = (
        args.als_params
    )

    # load the dataset for the other models
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

    # set up the recommenders
    cf_recommender = (SparseItemKNN(**cf_configurations)).fit(train)
    cbm_recommender = ContentRecommender(confidence_alpha=args.cbm_params["alpha"])
    cbm_recommender.fit(train_interactions, features)
    multi_cbm_recommender = MultiInterestContentRecommender(
        confidence_alpha=args.multicbm_params["alpha"], 
        global_weight=args.multicbm_params["global_weight"]
        )
    multi_cbm_recommender.fit(train_interactions, features=features, k=args.multicbm_params["k_for_kmeans"])
    als_recommender = ImplicitALS(**als_configurations).fit(train)

    results: dict[str, dict[str, float | int]] = defaultdict(dict)
    cf_seen = None
    # cf and als recommendations
    for recommender, label in zip([cf_recommender, als_recommender], ["cf", "als"]):
        recommendations = recommend_users(recommender, users, max_k)
        seen = {
            user: {str(recommender.data.item_ids[i]) for i in recommender.data.confidence.getrow(
                recommender.data.user_to_index[user]
            ).indices}
            for user in users if user in recommender.data.user_to_index
        }
        if label == "cf":
            cf_seen = seen
        assert_unseen_recommendations(recommendations, seen)
        metrics = evaluate_topk(recommendations, truth, set(map(str, recommender.data.item_ids)), ks)
        results[label] = metrics
        print(f"{label} metrics: {metrics}")

    # cbm and multi-cbm recommendations
    cbm_seen = cbm_recommender.seen.copy()
    cbm_catalog = cbm_recommender.catalog
    multi_cbm_seen = multi_cbm_recommender.seen.copy()
    multi_cbm_catalog = multi_cbm_recommender.catalog

    multi_cbm_recommendations = {}
    cbm_recommendations = {}
    for user in users:
        cbm_recommendations[user] = cbm_recommender.recommend(user, k=max_k)
        multi_cbm_recommendations[user] = multi_cbm_recommender.recommend(user, k=max_k)
    assert_unseen_recommendations(cbm_recommendations, cbm_seen)
    assert_unseen_recommendations(multi_cbm_recommendations, multi_cbm_seen)

    cbm_metrics = evaluate_topk(cbm_recommendations, truth=truth, catalog=cbm_catalog, k_values=ks)
    multi_cbm_metrics = evaluate_topk(multi_cbm_recommendations, truth=truth, catalog=multi_cbm_catalog, k_values=ks)
    results["cbm"] = cbm_metrics
    results["multi_cbm"] = multi_cbm_metrics
    print(f"cbm metrics: {cbm_metrics} \nmulti-interest cbm metrics: {multi_cbm_metrics}")

    # hybrid recommendations
    cf_weight = args.hybrid_params["cf_weight"]

    common_items = sorted(set(cf_recommender.data.item_ids) & set(cbm_recommender.catalog))
    n_items = len(common_items)

    cf_item_pos = np.array([cf_recommender.data.item_to_index[item] for item in common_items])
    cf_matrix = np.zeros((len(users), n_items), dtype=np.float32)
    cbm_matrix = np.zeros((len(users), n_items), dtype=np.float32)
    hybrid_seen_mask = np.zeros((len(users), n_items), dtype=bool)

    for row, user in enumerate(users):
        full_cf_scores = cf_recommender.score_not_processed(user) 
        cf_matrix[row] = full_cf_scores[cf_item_pos]

        cbm_scores = cbm_recommender.score(user)
        cbm_matrix[row] = [cbm_scores.get(item, 0.0) for item in common_items]

        user_hybrid_seen = cf_seen.get(user, set()) | cbm_seen.get(user, set())
        hybrid_seen_mask[row] = [item in user_hybrid_seen for item in common_items]

    cf_matrix = normalize_rows(cf_matrix)
    cbm_matrix = normalize_rows(cbm_matrix)

    combined = cf_weight * cf_matrix + (1 - cf_weight) * cbm_matrix
    combined[hybrid_seen_mask] = -np.inf 

    hybrid_recommendations = {}
    for row, user in enumerate(users):
        top_idx = np.argpartition(combined[row], -max_k)[-max_k:]
        top_idx = top_idx[np.argsort(-combined[row, top_idx])]
        hybrid_recommendations[user] = [common_items[i] for i in top_idx if np.isfinite(combined[row, i])]

    assert_unseen_recommendations(hybrid_recommendations, cf_seen)
    assert_unseen_recommendations(hybrid_recommendations, dict(cbm_recommender.seen))
        
    hybrid_metrics = evaluate_topk(hybrid_recommendations, truth, set(common_items), ks)
    results["hybrid"] = hybrid_metrics
    print(f"hybrid metrics: {hybrid_metrics}")

    output_version = args.output_version or "test_v1"
    output_dir = args.output_dir
    output_result = {
        "output_version": output_version,
        "split_version": args.split_version,
        "dataset_version": args.dataset_version,
        "feature_schema_version": args.feature_schema_version,
        "evaluation_split": "test",
        "metric_results": results
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{output_version}.json"
    output.write_text(json.dumps(output_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"End of the test")
    return results


def plot_main(results, config=False, save=True, baseline=None):
    args = parse_args()
    ks = args.k
    configutations = {
        "cf": args.cf_params,
        "als": args.als_params,
        "cbm": args.cbm_params,
        "multi_cbm": args.multicbm_params,
        "hybrid": args.hybrid_params
    }
    for k in ks:
        metric_names = [f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}", f"catalog_coverage@{k}"]
        if save: 
            output_dir = Path(OUTPUT_DIR / f"test_metrics_plot{k}.jpg")
        plot_result_bars(results=results, configurations=configutations, 
                         metric_names=metric_names, label_config=config, 
                         output_dir=output_dir, baseline=baseline)


if __name__ == "__main__":
    output_dir = OUTPUT_DIR / f"{OUTPUT_VERSION}.json"
    baseline_dir = BASELINE_DIR

    baseline_results = None
    if baseline_dir.exists():
        with open(baseline_dir, 'r') as f:
            baseline_data = json.load(f)
            baseline_results = baseline_data['models']

    if output_dir.exists():
        print("plotting from existing results")
        with open(output_dir, 'r') as f:
            data = json.load(f)
            results = data['metric_results']
            plot_main(results=results, save=True, config=False, baseline=baseline_results)
    else:
        print("no previous results, a new test starts")
        results = dict(main())
        plot_main(results, save=True, config=False, baseline=baseline_results)
