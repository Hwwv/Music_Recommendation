#!/usr/bin/env python3
"""Run the complete experiment interface on deterministic synthetic music data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from music_recommender.data import Interaction, leave_largest_out_split
from music_recommender.metrics import catalog_coverage, intra_list_diversity, ndcg_at_k, recall_at_k
from music_recommender.models import ContentRecommender, HybridRecommender, ItemKNN, MatrixFactorization, PopularityRecommender


def main() -> None:
    features = {
        "piano_1": [1.0, 0.0, 0.1], "piano_2": [0.9, 0.1, 0.1],
        "rock_1": [0.0, 1.0, 0.6], "rock_2": [0.1, 0.9, 0.7],
        "dance_1": [0.1, 0.4, 1.0], "dance_2": [0.0, 0.3, 0.9],
    }
    interactions = [
        Interaction("u1", "piano_1", 12), Interaction("u1", "rock_1", 4), Interaction("u1", "piano_2", 9),
        Interaction("u2", "piano_1", 8), Interaction("u2", "piano_2", 10), Interaction("u2", "dance_1", 2),
        Interaction("u3", "rock_1", 11), Interaction("u3", "rock_2", 9), Interaction("u3", "dance_2", 3),
        Interaction("u4", "dance_1", 10), Interaction("u4", "dance_2", 12), Interaction("u4", "rock_2", 2),
    ]
    train, truth = leave_largest_out_split(interactions, minimum_to_split=3, seed=1)
    popularity = PopularityRecommender().fit(train)
    knn = ItemKNN(neighbours=4).fit(train)
    mf = MatrixFactorization(factors=6, epochs=40).fit(train)
    content = ContentRecommender().fit(train, features)
    models = {
        "popularity": popularity,
        "item_knn": knn,
        "matrix_factorization": mf,
        "content": content,
        "hybrid_knn_content": HybridRecommender(knn, content, cf_weight=0.6),
    }
    print(f"train_interactions={len(train)} evaluated_users={len(truth)}")
    print("model                    recall@3  ndcg@3  coverage@3  diversity@3")
    for name, model in models.items():
        recs = {user: model.recommend(user, 3) for user in truth}
        print(f"{name:24} {recall_at_k(recs, truth, 3):8.3f} {ndcg_at_k(recs, truth, 3):7.3f} "
              f"{catalog_coverage(recs, set(features), 3):11.3f} {intra_list_diversity(recs, features, 3):12.3f}")


if __name__ == "__main__":
    main()

