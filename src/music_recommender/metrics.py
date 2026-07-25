"""Top-k relevance and beyond-accuracy metrics."""

from __future__ import annotations

import math
from typing import Mapping


def recall_at_k(recommendations: Mapping[str, list[str]], truth: Mapping[str, str], k: int) -> float:
    users = [u for u in truth if u in recommendations]
    return sum(truth[u] in recommendations[u][:k] for u in users) / len(users) if users else 0.0


def ndcg_at_k(recommendations: Mapping[str, list[str]], truth: Mapping[str, str], k: int) -> float:
    users = [u for u in truth if u in recommendations]
    if not users:
        return 0.0
    total = 0.0
    for user in users:
        try:
            rank = recommendations[user][:k].index(truth[user]) + 1
            total += 1.0 / math.log2(rank + 1)
        except ValueError:
            pass
    return total / len(users)

def ndcg_at_k_multiple(recommendations: Mapping[str, list[str]], truth: Mapping[str, list[str]], k: int) -> float:
    users = [u for u in truth if u in recommendations]
    if not users:
        return 0.0
    total = 0.0
    for user in users:
        recs = recommendations[user][:k]

        dcg = 0.0
        for rank, item in enumerate(recs):
            if item in truth[user]:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0/math.log2(rank + 1) for rank in range(min(k, len(truth[user]))))
        if idcg > 0:
            total += dcg / idcg

    return total / len(users)


def catalog_coverage(recommendations: Mapping[str, list[str]], catalog: set[str], k: int) -> float:
    exposed = {item for values in recommendations.values() for item in values[:k]}
    return len(exposed) / len(catalog) if catalog else 0.0


def intra_list_diversity(recommendations: Mapping[str, list[str]], features: Mapping[str, list[float]], k: int) -> float:
    def cosine(a: list[float], b: list[float]) -> float:
        denom = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
        return sum(x*y for x, y in zip(a, b)) / denom if denom else 0.0
    values: list[float] = []
    for ranked in recommendations.values():
        items = [item for item in ranked[:k] if item in features]
        pairs = [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
        if pairs:
            values.append(sum(1.0 - cosine(features[a], features[b]) for a, b in pairs) / len(pairs))
    return sum(values) / len(values) if values else 0.0

