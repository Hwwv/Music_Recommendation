#!/usr/bin/env python3
"""Diagnose how play history spreads across content clusters.

Run from the project root:
    python scripts/diagnose_clusters.py --k 10 --alpha 0.8
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from music_recommender.data_loader import MusicDataLoader
from music_recommender.models import MultiInterestContentRecommender

INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
OUTPUT_DIR = ROOT / "artifacts" / "diagnostics"

DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_version", default=DATASET_VERSION)
    parser.add_argument("--split_version", default=SPLIT_VERSION)
    parser.add_argument("--feature_schema_version", default=FEATURE_SCHEMA_VERSION)
    parser.add_argument("--k", type=int, default=10, help="KMeans cluster count to diagnose")
    parser.add_argument("--alpha", type=float, default=0.8, help="confidence_alpha for weighting")
    parser.add_argument("--min_history", type=int, default=15, help="min train interactions to qualify as an example user")
    parser.add_argument("--user_id", type=int, default=None, help="force a specific user_id for the example plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_loader = MusicDataLoader(
        data_version=args.dataset_version,
        split_version=args.split_version,
        feature_schema_version=args.feature_schema_version,
        data_db_path=INTEGRATION,
        allow_test=False,
    )

    train_interactions = data_loader.load_split_interactions("train")
    features = data_loader.load_feature_mappings()

    model = MultiInterestContentRecommender(confidence_alpha=args.alpha, global_weight=0.1)
    model.fit(train_interactions, features, k=args.k)

    # distribution of number of distinct clusters touched per user
    user_clusters_touched: dict[int, set[int]] = {}
    for user, hist in model.history.items():
        clusters = {model.cluster_labels[item] for item, _ in hist if item in model.cluster_labels}
        user_clusters_touched[user] = clusters

    n_clusters_per_user = np.array([len(c) for c in user_clusters_touched.values()])
    n_items_per_user = np.array([len(model.history[u]) for u in user_clusters_touched])

    print(f"k (clusters) = {args.k}")
    print(f"users with train history: {len(n_clusters_per_user)}")
    print(f"mean # distinct clusters touched: {n_clusters_per_user.mean():.2f}")
    print(f"median # distinct clusters touched: {np.median(n_clusters_per_user):.0f}")
    print(f"users touching exactly 1 cluster: {(n_clusters_per_user == 1).sum()} "
          f"({(n_clusters_per_user == 1).mean() * 100:.1f}%)")
    print(f"users touching >= 2 clusters: {(n_clusters_per_user >= 2).sum()} "
          f"({(n_clusters_per_user >= 2).mean() * 100:.1f}%)")
    print(f"users touching >= 3 clusters: {(n_clusters_per_user >= 3).sum()} "
          f"({(n_clusters_per_user >= 3).mean() * 100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    max_c = int(n_clusters_per_user.max())
    ax.hist(n_clusters_per_user, bins=np.arange(0.5, max_c + 1.5, 1), edgecolor="black")
    ax.set_xlabel("Distinct clusters touched by a user's train history")
    ax.set_ylabel("Number of users")
    ax.set_title(f"Cluster diversity per user (k={args.k})")
    fig.tight_layout()
    hist_path = OUTPUT_DIR / f"user_cluster_diversity_hist_k{args.k}.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # also relate diversity to history length, since a user with 5 items can't
    # touch more than 5 clusters -- normalize to see if it's a real signal
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n_items_per_user, n_clusters_per_user, alpha=0.15, s=10)
    ax.set_xlabel("# train interactions")
    ax.set_ylabel("# distinct clusters touched")
    ax.set_title(f"Cluster diversity vs. history length (k={args.k})")
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / f"cluster_diversity_vs_history_len_k{args.k}.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {scatter_path}")

    # pick / use one example user and visualize their history
    if args.user_id is not None:
        example_user = args.user_id
        if example_user not in model.history:
            raise SystemExit(f"user_id {example_user} has no train history under this split")
    else:
        candidates = [
            (u, len(c)) for u, c in user_clusters_touched.items()
            if len(model.history[u]) >= args.min_history
        ]
        if not candidates:
            raise SystemExit(f"no users with >= {args.min_history} train interactions; lower --min_history")
        candidates.sort(key=lambda x: -x[1])
        # pick a user around the 25th percentile of diversity (interesting, not an outlier)
        example_user = candidates[len(candidates) // 4][0]

    hist = model.history[example_user]
    items = [item for item, _ in hist]
    weights = [w for _, w in hist]
    clusters = [model.cluster_labels[item] for item in items]
    print(f"\nExample user: {example_user}")
    print(f"  # train interactions: {len(items)}")
    print(f"  # distinct clusters touched: {len(set(clusters))} / {args.k}")

    # bar chart: total confidence weight per cluster for this user
    cluster_weight: dict[int, float] = defaultdict(float)
    for c, w in zip(clusters, weights):
        cluster_weight[c] += w
    sorted_clusters = sorted(cluster_weight.items())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(c) for c, _ in sorted_clusters], [w for _, w in sorted_clusters])
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Total confidence weight")
    ax.set_title(f"User {example_user}: play history weight by cluster (k={args.k})")
    fig.tight_layout()
    bar_path = OUTPUT_DIR / f"user_{example_user}_cluster_weights_k{args.k}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # PCA scatter of the whole catalog, colored by cluster, user's items highlighted
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=311)
    coords = pca.fit_transform(model.feature_matrix)
    item_index = {item: i for i, item in enumerate(model.item_ids)}
    all_cluster_labels = np.array([model.cluster_labels[item] for item in model.item_ids])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=all_cluster_labels, cmap="tab20", s=6, alpha=0.25)
    user_idx = [item_index[item] for item in items if item in item_index]
    max_w = max(weights) if weights else 1.0
    user_sizes = [30 + 220 * (w / max_w) for w in weights]
    ax.scatter(
        coords[user_idx, 0], coords[user_idx, 1],
        c="red", edgecolors="black", s=user_sizes,
        label=f"User {example_user} history", zorder=5,
    )
    ax.set_ylim(bottom=-8)
    ax.set_title(f"Item feature space (PCA), user {example_user} history highlighted (k={args.k})")
    ax.legend()
    fig.tight_layout()
    scatter2_path = OUTPUT_DIR / f"user_{example_user}_pca_scatter_k{args.k}.png"
    fig.savefig(scatter2_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {scatter2_path}")


if __name__ == "__main__":
    main()