"""Small, readable recommender baselines for the first project milestone.

The implementations prioritize experimental transparency. They are suitable for
the smoke dataset and medium-sized sampled experiments; the full 500K-user run
will later replace the internals with sparse-array implementations while keeping
the same fit/score/recommend interface.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable, Mapping

from .data import Interaction


ScoreMap = dict[str, float]


def _cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0


def _normalize(scores: Mapping[str, float]) -> ScoreMap:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


class BaseRecommender:
    catalog: set[str]
    seen: dict[str, set[str]]

    def score(self, user_id: str) -> ScoreMap:
        raise NotImplementedError

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        seen = self.seen.get(user_id, set())
        scores = self.score(user_id)
        return [item for item, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0])) if item not in seen][:k]


class PopularityRecommender(BaseRecommender):
    def fit(self, interactions: Iterable[Interaction]) -> "PopularityRecommender":
        totals: dict[str, float] = defaultdict(float)
        self.seen = defaultdict(set)
        for row in interactions:
            totals[row.item_id] += math.log1p(max(0.0, row.play_count))
            self.seen[row.user_id].add(row.item_id)
        self.totals = dict(totals)
        self.catalog = set(totals)
        return self

    def score(self, user_id: str) -> ScoreMap:
        return dict(self.totals)


class ItemKNN(BaseRecommender):
    """Implicit item-item cosine KNN with play-count confidence weights."""

    def __init__(self, neighbours: int = 30, confidence_alpha: float = 1.0):
        self.neighbours = neighbours
        self.confidence_alpha = confidence_alpha

    def fit(self, interactions: Iterable[Interaction]) -> "ItemKNN":
        self.user_weights: dict[str, dict[str, float]] = defaultdict(dict)
        item_users: dict[str, dict[str, float]] = defaultdict(dict)
        self.seen = defaultdict(set)
        for row in interactions:
            weight = 1.0 + self.confidence_alpha * math.log1p(max(0.0, row.play_count))
            self.user_weights[row.user_id][row.item_id] = weight
            item_users[row.item_id][row.user_id] = weight
            self.seen[row.user_id].add(row.item_id)
        self.catalog = set(item_users)
        norms = {item: math.sqrt(sum(v * v for v in users.values())) for item, users in item_users.items()}
        self.similar: dict[str, list[tuple[str, float]]] = {}
        items = sorted(self.catalog)
        for item in items:
            sims: list[tuple[str, float]] = []
            for other in items:
                if other == item:
                    continue
                common = set(item_users[item]) & set(item_users[other])
                dot = sum(item_users[item][u] * item_users[other][u] for u in common)
                denom = norms[item] * norms[other]
                if dot and denom:
                    sims.append((other, dot / denom))
            self.similar[item] = sorted(sims, key=lambda x: (-x[1], x[0]))[: self.neighbours]
        return self

    def score(self, user_id: str) -> ScoreMap:
        scores: dict[str, float] = defaultdict(float)
        for item, weight in self.user_weights.get(user_id, {}).items():
            for candidate, similarity in self.similar.get(item, []):
                scores[candidate] += weight * similarity
        return dict(scores)


class MatrixFactorization(BaseRecommender):
    """Confidence-weighted latent factor model trained with deterministic SGD."""

    def __init__(self, factors: int = 12, epochs: int = 25, learning_rate: float = 0.03, regularization: float = 0.02, confidence_alpha: float = 1.0, seed: int = 311):
        self.factors, self.epochs = factors, epochs
        self.learning_rate, self.regularization = learning_rate, regularization
        self.confidence_alpha, self.seed = confidence_alpha, seed

    def fit(self, interactions: Iterable[Interaction]) -> "MatrixFactorization":
        rows = list(interactions)
        users = sorted({row.user_id for row in rows})
        items = sorted({row.item_id for row in rows})
        self.catalog, self.seen = set(items), defaultdict(set)
        rng = random.Random(self.seed)
        self.user_vec = {u: [rng.uniform(-0.1, 0.1) for _ in range(self.factors)] for u in users}
        self.item_vec = {i: [rng.uniform(-0.1, 0.1) for _ in range(self.factors)] for i in items}
        examples: list[tuple[str, str, float]] = []
        for row in rows:
            self.seen[row.user_id].add(row.item_id)
            confidence = 1.0 + self.confidence_alpha * math.log1p(max(0.0, row.play_count))
            examples.append((row.user_id, row.item_id, confidence))
        for _ in range(self.epochs):
            rng.shuffle(examples)
            for user, item, confidence in examples:
                uv, iv = self.user_vec[user], self.item_vec[item]
                prediction = sum(a * b for a, b in zip(uv, iv))
                error = confidence * (1.0 - prediction)
                for f in range(self.factors):
                    old_u = uv[f]
                    uv[f] += self.learning_rate * (error * iv[f] - self.regularization * uv[f])
                    iv[f] += self.learning_rate * (error * old_u - self.regularization * iv[f])
        return self

    def score(self, user_id: str) -> ScoreMap:
        if user_id not in self.user_vec:
            return {}
        uv = self.user_vec[user_id]
        return {item: sum(a * b for a, b in zip(uv, vec)) for item, vec in self.item_vec.items()}


class ContentRecommender(BaseRecommender):
    def __init__(self, confidence_alpha: float = 1.0):
        self.confidence_alpha = confidence_alpha

    def fit(self, interactions: Iterable[Interaction], features: Mapping[str, list[float]]) -> "ContentRecommender":
        self.features = {item: list(vector) for item, vector in features.items()}
        self.catalog = set(self.features)
        self.history: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.seen = defaultdict(set)
        for row in interactions:
            if row.item_id not in self.features:
                continue
            weight = 1.0 + self.confidence_alpha * math.log1p(max(0.0, row.play_count))
            self.history[row.user_id].append((row.item_id, weight))
            self.seen[row.user_id].add(row.item_id)
        return self

    def score(self, user_id: str) -> ScoreMap:
        history = self.history.get(user_id, [])
        if not history:
            return {}
        width = len(next(iter(self.features.values())))
        total = sum(weight for _, weight in history)
        profile = [sum(self.features[item][j] * weight for item, weight in history) / total for j in range(width)]
        return {item: _cosine(profile, vector) for item, vector in self.features.items()}


class HybridRecommender(BaseRecommender):
    def __init__(self, collaborative: BaseRecommender, content: BaseRecommender, cf_weight: float = 0.6):
        if not 0.0 <= cf_weight <= 1.0:
            raise ValueError("cf_weight must be between 0 and 1")
        self.collaborative, self.content, self.cf_weight = collaborative, content, cf_weight
        self.catalog = collaborative.catalog | content.catalog
        self.seen = defaultdict(set)
        for user in set(collaborative.seen) | set(content.seen):
            self.seen[user] = collaborative.seen.get(user, set()) | content.seen.get(user, set())

    def score(self, user_id: str) -> ScoreMap:
        cf, cb = _normalize(self.collaborative.score(user_id)), _normalize(self.content.score(user_id))
        return {item: self.cf_weight * cf.get(item, 0.0) + (1.0 - self.cf_weight) * cb.get(item, 0.0) for item in self.catalog}

