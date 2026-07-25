"""Data contracts and leakage-safe splitting utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
import re
import unicodedata


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    play_count: float = 1.0


def normalize_text(value: str) -> str:
    """Normalize a track or artist field for conservative exact matching."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def track_key(track_name: str, artist_name: str) -> tuple[str, str]:
    return normalize_text(track_name), normalize_text(artist_name)


def leave_largest_out_split(
    interactions: list[Interaction], minimum_to_split: int = 5, seed: int = 311
) -> tuple[list[Interaction], dict[str, str]]:
    """Hold out the largest-playcount item per eligible user.

    Users with fewer than [minimum_to_split] distinct items remain entirely in training because
    withholding their largest observation would cause significant bias in evaluation.
    """
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row.user_id].append(row)

    rng = random.Random(seed)
    train: list[Interaction] = []
    test: dict[str, str] = {}
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        unique_items = sorted({row.item_id for row in rows})
        if len(unique_items) < minimum_to_split:
            train.extend(rows)
            continue
        held_out_row = max(rows, key=lambda r: (r.play_count, rng.random()))
        test[user_id] = held_out_row.item_id
        train.extend(row for row in rows if row.item_id != held_out_row.item_id)
    return train, test


def interaction_split(
        interactions: list[Interaction], frac: float = 0.2, n: int = 1, minimum_to_split: int = 5, seed: int = 311, type: str = 'frac'
) -> tuple[list[Interaction], dict[str, list[str]]]:
    """Hold out a fraction or number of items per eligible user, given the input parameter [type].

    Users with fewer than [minimum_to_split] distinct items remain entirely in training because
    withholding their observation would cause significant bias in evaluation.
    """
    assert type in ['frac', 'n'], "type must be either 'frac' or 'n'"

    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row.user_id].append(row)

    rng = random.Random(seed)
    train: list[Interaction] = []
    test: dict[str, list[str]] = {}
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        unique_items = sorted({row.item_id for row in rows})
        if len(unique_items) < minimum_to_split:
            train.extend(rows)
            continue
        if type == 'frac':
            n_holdout = max(1, min(int(len(unique_items) * frac), len(unique_items) - 1))
        elif type == 'n':
            n_holdout = min(n, len(unique_items) - 1)
        held_out_rows = rng.sample(unique_items, n_holdout)
        test[user_id] = held_out_rows
        train.extend(row for row in rows if row.item_id not in held_out_rows)
    return train, test