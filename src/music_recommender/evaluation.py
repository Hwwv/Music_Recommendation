"""Shared top-k evaluation for implicit-feedback recommendation experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set


def evaluate_topk(
    recommendations: Mapping[int | str, Sequence[int | str]],
    truth: Mapping[int | str, Set[int | str]],
    catalog: Set[int | str],
    k_values: Sequence[int] = (10, 20),
) -> dict[str, float | int]:
    """Evaluate ranked recommendations against multiple held-out items per user."""
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    users = sorted(set(truth) & set(recommendations), key=str)
    results: dict[str, float | int] = {"evaluated_users": len(users)}
    for k in sorted(set(k_values)):
        recall_total = ndcg_total = hit_total = 0.0
        exposed: set[int | str] = set()
        for user in users:
            relevant = truth[user]
            ranked = list(recommendations[user][:k])
            exposed.update(ranked)
            hits = [rank for rank, item in enumerate(ranked, start=1) if item in relevant]
            recall_total += len(hits) / len(relevant) if relevant else 0.0
            hit_total += float(bool(hits))
            dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
            ideal_hits = min(k, len(relevant))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_total += dcg / idcg if idcg else 0.0
        denominator = len(users)
        results[f"recall@{k}"] = recall_total / denominator if denominator else 0.0
        results[f"ndcg@{k}"] = ndcg_total / denominator if denominator else 0.0
        results[f"hit_rate@{k}"] = hit_total / denominator if denominator else 0.0
        results[f"catalog_coverage@{k}"] = len(exposed) / len(catalog) if catalog else 0.0
        results[f"exposed_items@{k}"] = len(exposed)
    return results


def assert_unseen_recommendations(
    recommendations: Mapping[int | str, Sequence[int | str]],
    seen: Mapping[int | str, Set[int | str]],
) -> None:
    """Raise if any model recommends an item from that user's training history."""
    for user, ranked in recommendations.items():
        overlap = set(ranked) & seen.get(user, set())
        if overlap:
            raise AssertionError(f"user {user!r} received seen items: {sorted(overlap, key=str)[:5]}")
