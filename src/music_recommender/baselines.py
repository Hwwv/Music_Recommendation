"""Scalable deterministic baselines for the versioned full-data experiments."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence, Set


def recommend_from_ranking(
    users: Sequence[int],
    seen: Mapping[int, Set[str]],
    ranking: Sequence[str],
    k: int,
) -> dict[int, list[str]]:
    recommendations: dict[int, list[str]] = {}
    for user in users:
        user_seen = seen.get(user, set())
        recommendations[user] = [item for item in ranking if item not in user_seen][:k]
    return recommendations


def recommend_history_popularity(
    users: Sequence[int],
    seen: Mapping[int, Set[str]],
    target_popularity: Mapping[int, int],
    rankings_by_popularity: Mapping[int, Sequence[str]],
    k: int,
) -> dict[int, list[str]]:
    recommendations: dict[int, list[str]] = {}
    for user in users:
        ranking = rankings_by_popularity[target_popularity[user]]
        user_seen = seen.get(user, set())
        recommendations[user] = [item for item in ranking if item not in user_seen][:k]
    return recommendations


def recommend_random(
    users: Sequence[int],
    seen: Mapping[int, Set[str]],
    catalog: Sequence[str],
    k: int,
    seed: int,
) -> dict[int, list[str]]:
    """Sample unseen items without replacement using a stable per-user seed."""
    recommendations: dict[int, list[str]] = {}
    for user in users:
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{user}".encode()).digest()[:8], "big"
        )
        rng = random.Random(stable_seed)
        user_seen = seen.get(user, set())
        selected: list[str] = []
        selected_set: set[str] = set()
        while len(selected) < k:
            item = catalog[rng.randrange(len(catalog))]
            if item not in user_seen and item not in selected_set:
                selected.append(item)
                selected_set.add(item)
        recommendations[user] = selected
    return recommendations
