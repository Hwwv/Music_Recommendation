"""Run the Content and Multi-Interest Recommenders (default data version v1)"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import sys
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from music_recommender.data_loader import MusicDataLoader
from music_recommender.evaluation import evaluate_topk, assert_unseen_recommendations
from music_recommender.models import ContentRecommender

INTEGRATION = ROOT / "data" / "databases2" / "integration.duckdb"
CBM_OUTPUT_DIR = ROOT / "artifacts2" / "cbm"

DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"
CBM_OUTPUT_VERSION = "cbm_eval20_validation_v1"

ALLOW_TEST = False 
KS = [10, 20]
ALPHAS = [0.7, 0.8, 0.9, 0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Content and Multi-Interest Recommenders")
    parser.add_argument("--dataset_version", type=str, default=DATASET_VERSION, help="Dataset version to use")
    parser.add_argument("--split_version", type=str, default=SPLIT_VERSION, help="Split version to use")
    parser.add_argument("--feature_schema_version", type=str, default=FEATURE_SCHEMA_VERSION, help="Feature schema version to use")
    parser.add_argument("--data_db_path", type=Path, default=INTEGRATION, help="Path to the DuckDB database file")
    parser.add_argument("--cbm_output_version", type=str, default=CBM_OUTPUT_VERSION, help="Output version for CBM results")
    parser.add_argument("--cbm_output_dir", type=str, default=CBM_OUTPUT_DIR, help="Output directory for CBM results")
    parser.add_argument("--allow_test", action="store_true", help="Allow loading of test split (default: False)")
    parser.add_argument("--ks", type=int, nargs="+", default=KS, help="List of k values for evaluation metrics")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS, help="List of alpha values for the Content-Based Recommender")
    return parser.parse_args()


def plot_cbm_experiments(
    result: Dict[float, Dict[str, Any]],
    ks: List[int],
    alphas: List[float],
    output_dir: Path,
    plot_name: str = "cbm_hyperparameter_search"
) -> None:
    """
    Plot CBM hyperparameter search results.
    
    Args:
        result: Dictionary mapping alpha -> metrics
        ks: List of k values
        alphas: List of alpha values
        output_dir: Directory to save plots
        plot_name: Base name for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for alpha in alphas:
        if alpha not in result:
            raise ValueError(f"Incomplete result: alpha {alpha} missing.")
  
    metrics_to_plot = ["recall", "ndcg", "hit_rate", "catalog_coverage"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ks)))

    # line plots for each k with respect to alpha
    for k in ks:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = []
            for alpha in alphas:
                values.append(result[alpha][f"{metric}@{k}"])
            
            ax.plot(alphas, values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Alpha (confidence_alpha)', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} @ {k}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            best_idx = np.argmax(values)
            ax.plot(alphas[best_idx], values[best_idx], 'r*', markersize=15, 
                   label=f'Best: {values[best_idx]:.4f} at alpha={alphas[best_idx]:.2f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{plot_name}_metrics_k{k}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / f'{plot_name}_metrics_k{k}.png'}")
    
    # For each metric, compare different ks
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, k in enumerate(ks):
            values = []
            for alpha in alphas:
                values.append(result[alpha][f"{metric}@{k}"])
            ax.plot(alphas, values, marker='o', linewidth=2, markersize=8,
                   color=colors[idx], label=f'K={k}')
        
        ax.set_xlabel('Alpha (confidence_alpha)', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Alpha for Different K', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{plot_name}_{metric}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / f'{plot_name}_{metric}_comparison.png'}")
    
    # Heatmap for metrics and alphas for each k
    for k in ks:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        heatmap_data = []
        for alpha in alphas:
            row = []
            for metric in metrics_to_plot:
                row.append(result[alpha][f"{metric}@{k}"])
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(metrics_to_plot)))
        ax.set_yticks(np.arange(len(alphas)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.set_yticklabels([f'{a:.2f}' for a in alphas])
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(alphas)):
            for j in range(len(metrics_to_plot)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="black" if heatmap_data[i, j] < 0.5 else "white")
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Alpha', fontsize=12)
        ax.set_title(f'Performance Heatmap (K={k})', fontsize=14)
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(output_dir / f"{plot_name}_heatmap_k{k}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / f'{plot_name}_heatmap_k{k}.png'}")

    # best alphas
    print("\n" + "="*60)
    print("Best Alpha Summary:")
    print("="*60)
    for k in ks:
        best_alpha = None
        best_recall = -1
        for alpha in alphas:
            recall = result[alpha][f"recall@{k}"]
            if recall > best_recall:
                best_recall = recall
                best_alpha = alpha
        print(f"K={k:2d} | Best alpha: {best_alpha:.2f} | Recall@{k}: {best_recall:.4f}")
    


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

    # Train the models with different hyperparameters and get results for the validation set
    results_cbm: dict[float, dict] = defaultdict(dict)

    train_interactions = data_loader.load_split_interactions("train")
    features = data_loader.load_feature_mappings()

    validation_truths = data_loader.load_split_truth("validation")
    validation_users = sorted(set(validation_truths.keys()))

    # Train and evaluate the Content-Based Recommender (CBM)
    for alpha in args.alphas:
        print(f"alpha={alpha}")
        cbm_recommender = ContentRecommender(confidence_alpha=alpha)
        cbm_recommender.fit(train_interactions, features)

        seen = cbm_recommender.seen.copy()
        catalog = cbm_recommender.catalog

        recommendations = {}
        for user in validation_users:
            recommendations[user] = cbm_recommender.recommend(user, k=max(args.ks))
        assert_unseen_recommendations(recommendations, seen)

        metrics = evaluate_topk(recommendations, validation_truths, catalog, k_values=args.ks)
        results_cbm[alpha] = metrics

    output_dir = Path(args.cbm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cbm_output = output_dir / f"{CBM_OUTPUT_VERSION}.json"
    cbm_output.write_text(json.dumps(results_cbm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CBM results saved to {cbm_output.resolve()}")

    plot_cbm_experiments(result=results_cbm, ks=args.ks, alphas=args.alphas, output_dir=args.cbm_output_dir)


if __name__ == "__main__":
    main()