"""Scalable deterministic baselines for the versioned full-data experiments."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence, Set


def recommend_from_ranking(
    users: Sequence[int],
    seen: Mapping[int, Set[int]],
    ranking: Sequence[int],
    k: int,
) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}
    for user in users:
        user_seen = seen.get(user, set())
        recommendations[user] = [item for item in ranking if item not in user_seen][:k]
    return recommendations


def recommend_history_popularity(
    users: Sequence[int],
    seen: Mapping[int, Set[int]],
    target_popularity: Mapping[int, int],
    rankings_by_popularity: Mapping[int, Sequence[int]],
    k: int,
) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}
    for user in users:
        ranking = rankings_by_popularity[target_popularity[user]]
        user_seen = seen.get(user, set())
        recommendations[user] = [item for item in ranking if item not in user_seen][:k]
    return recommendations


def recommend_random(
    users: Sequence[int],
    seen: Mapping[int, Set[int]],
    catalog: Sequence[int],
    k: int,
    seed: int,
    stable_user_ids: Mapping[int, int] | None = None,
) -> dict[int, list[int]]:
    """Sample unseen items without replacement using a stable per-user seed."""
    recommendations: dict[int, list[int]] = {}
    for user in users:
        seed_user = stable_user_ids[user] if stable_user_ids is not None else user
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{seed_user}".encode()).digest()[:8], "big"
        )
        rng = random.Random(stable_seed)
        user_seen = seen.get(user, set())
        selected: list[int] = []
        selected_set: set[int] = set()
        while len(selected) < k:
            item = catalog[rng.randrange(len(catalog))]
            if item not in user_seen and item not in selected_set:
                selected.append(item)
                selected_set.add(item)
        recommendations[user] = selected
    return recommendations
