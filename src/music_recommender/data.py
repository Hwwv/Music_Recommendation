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


def leave_one_out_split(
    interactions: list[Interaction], seed: int = 311
) -> tuple[list[Interaction], dict[str, str]]:
    """Hold out one listened item per eligible user.

    Users with fewer than two distinct items remain entirely in training because
    withholding their only observation would make personalized evaluation invalid.
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
        if len(unique_items) < 2:
            train.extend(rows)
            continue
        held_out = rng.choice(unique_items)
        test[user_id] = held_out
        train.extend(row for row in rows if row.item_id != held_out)
    return train, test

