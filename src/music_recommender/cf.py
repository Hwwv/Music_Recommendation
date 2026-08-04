"""Sparse collaborative-filtering models for implicit listening feedback.

The ALS objective follows Hu, Koren, and Volinsky (2008): binary preference
``p_ui`` is fitted with confidence ``c_ui = 1 + alpha * log(1 + playcount)``.
The item-KNN implementation computes similarities in item blocks, so it never
materializes a dense item-by-item (or user-by-user) similarity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy import sparse

Weighting = Literal["cosine", "bm25"]


@dataclass(frozen=True)
class InteractionMatrix:
    confidence: sparse.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_to_index: dict[int, int]
    item_to_index: dict[str, int]


def build_interaction_matrix(
    rows: Iterable[object], alpha: float = 1.0
) -> InteractionMatrix:
    """Build a CSR confidence matrix from Interaction objects or dataframe rows."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    triples = [
        (
            int(getattr(row, "user_id")),
            str(getattr(row, "item_id", getattr(row, "feature_cluster_id", ""))),
            float(getattr(row, "play_count", getattr(row, "playcount_raw", 1.0))),
        )
        for row in rows
    ]
    users = np.asarray(sorted({x[0] for x in triples}), dtype=np.int64)
    items = np.asarray(sorted({x[1] for x in triples}), dtype=object)
    user_to_index = {int(value): i for i, value in enumerate(users)}
    item_to_index = {str(value): i for i, value in enumerate(items)}
    row_idx = np.fromiter((user_to_index[u] for u, _, _ in triples), dtype=np.int64)
    col_idx = np.fromiter((item_to_index[i] for _, i, _ in triples), dtype=np.int64)
    values = np.fromiter(
        (1.0 + alpha * np.log1p(max(0.0, count)) for _, _, count in triples),
        dtype=np.float64,
    )
    matrix = sparse.coo_matrix(
        (values, (row_idx, col_idx)), shape=(len(users), len(items))
    ).tocsr()
    # A split should contain one row per pair; summing duplicate confidences
    # would incorrectly add the baseline 1 more than once.
    if matrix.nnz != len(triples):
        raise ValueError("duplicate user-item pairs found in training interactions")
    return InteractionMatrix(matrix, users, items, user_to_index, item_to_index)


def _bm25_weight(matrix: sparse.csr_matrix, k1: float, b: float) -> sparse.csr_matrix:
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    result = matrix.astype(np.float64, copy=True).tocsr()
    lengths = np.asarray(result.sum(axis=1)).ravel()
    average = lengths.mean() if lengths.size else 1.0
    df = np.diff(result.tocsc().indptr)
    # Positive, smoothed IDF is stable for very popular items.
    idf = np.log1p(result.shape[0] / (1.0 + df))
    row_scale = k1 * (1.0 - b + b * lengths / max(average, 1e-12))
    for row in range(result.shape[0]):
        start, end = result.indptr[row : row + 2]
        values = result.data[start:end]
        result.data[start:end] = (
            idf[result.indices[start:end]] * values * (k1 + 1.0) / (values + row_scale[row])
        )
    return result


class SparseItemKNN:
    def __init__(
        self,
        neighbours: int = 100,
        alpha: float = 1.0,
        weighting: Weighting = "cosine",
        min_cooccurrence: int = 1,
        block_size: int = 256,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        if neighbours <= 0 or min_cooccurrence <= 0 or block_size <= 0:
            raise ValueError("neighbours, min_cooccurrence, and block_size must be positive")
        if weighting not in ("cosine", "bm25"):
            raise ValueError("weighting must be 'cosine' or 'bm25'")
        self.neighbours, self.alpha, self.weighting = neighbours, alpha, weighting
        self.min_cooccurrence, self.block_size = min_cooccurrence, block_size
        self.bm25_k1, self.bm25_b = bm25_k1, bm25_b

    def fit(self, rows: Iterable[object]) -> "SparseItemKNN":
        self.data = build_interaction_matrix(rows, self.alpha)
        confidence = self.data.confidence
        weighted = confidence if self.weighting == "cosine" else _bm25_weight(
            confidence, self.bm25_k1, self.bm25_b
        )
        norms = np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).ravel())
        normalized = weighted @ sparse.diags(1.0 / np.maximum(norms, 1e-12))
        binary = confidence.copy()
        binary.data[:] = 1.0
        blocks: list[sparse.csr_matrix] = []
        item_user = normalized.T.tocsr()
        binary_item_user = binary.T.tocsr()
        for start in range(0, confidence.shape[1], self.block_size):
            stop = min(start + self.block_size, confidence.shape[1])
            similarities = (item_user[start:stop] @ normalized).tocsr()
            cooccurrence = (binary_item_user[start:stop] @ binary).tocsr()
            block_rows: list[int] = []
            block_cols: list[int] = []
            block_values: list[float] = []
            for local_row in range(stop - start):
                s0, s1 = similarities.indptr[local_row : local_row + 2]
                candidates = similarities.indices[s0:s1]
                values = similarities.data[s0:s1]
                c0, c1 = cooccurrence.indptr[local_row : local_row + 2]
                counts = dict(zip(cooccurrence.indices[c0:c1], cooccurrence.data[c0:c1]))
                keep = np.asarray(
                    [j != start + local_row and counts.get(int(j), 0) >= self.min_cooccurrence for j in candidates]
                )
                candidates, values = candidates[keep], values[keep]
                if values.size > self.neighbours:
                    top = np.argpartition(values, -self.neighbours)[-self.neighbours:]
                    candidates, values = candidates[top], values[top]
                order = np.lexsort((candidates, -values))
                candidates, values = candidates[order], values[order]
                block_rows.extend([local_row] * len(candidates))
                block_cols.extend(candidates.tolist())
                block_values.extend(values.tolist())
            blocks.append(sparse.csr_matrix(
                (block_values, (block_rows, block_cols)),
                shape=(stop - start, confidence.shape[1]),
            ))
        self.similarity = sparse.vstack(blocks, format="csr")
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[str]:
        user = self.data.user_to_index.get(int(user_id))
        if user is None or k <= 0:
            return []
        history = self.data.confidence.getrow(user)
        scores = (history @ self.similarity).toarray().ravel()
        scores[history.indices] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        take = min(k, len(valid))
        if not take:
            return []
        top = valid[np.argpartition(scores[valid], -take)[-take:]]
        top = top[np.lexsort((self.data.item_ids[top], -scores[top]))]
        return [str(self.data.item_ids[i]) for i in top]


class ImplicitALS:
    """Exact alternating least squares for the confidence-weighted implicit objective."""

    def __init__(
        self, factors: int = 64, regularization: float = 0.01,
        iterations: int = 15, alpha: float = 1.0, seed: int = 42,
    ):
        if factors <= 0 or regularization < 0 or iterations <= 0:
            raise ValueError("invalid ALS hyperparameters")
        self.factors, self.regularization = factors, regularization
        self.iterations, self.alpha, self.seed = iterations, alpha, seed

    @staticmethod
    def _least_squares(
        interactions: sparse.csr_matrix, fixed: np.ndarray, regularization: float
    ) -> np.ndarray:
        result = np.empty((interactions.shape[0], fixed.shape[1]), dtype=np.float64)
        gram = fixed.T @ fixed
        identity = np.eye(fixed.shape[1])
        for entity in range(interactions.shape[0]):
            start, end = interactions.indptr[entity : entity + 2]
            indices = interactions.indices[start:end]
            confidence = interactions.data[start:end]
            selected = fixed[indices]
            extra = confidence - 1.0
            lhs = gram + (selected.T * extra) @ selected + regularization * identity
            rhs = selected.T @ confidence
            result[entity] = np.linalg.solve(lhs, rhs)
        return result

    def fit(self, rows: Iterable[object]) -> "ImplicitALS":
        self.data = build_interaction_matrix(rows, self.alpha)
        rng = np.random.default_rng(self.seed)
        users = rng.normal(0, 0.01, (self.data.confidence.shape[0], self.factors))
        items = rng.normal(0, 0.01, (self.data.confidence.shape[1], self.factors))
        for _ in range(self.iterations):
            users = self._least_squares(self.data.confidence, items, self.regularization)
            items = self._least_squares(self.data.confidence.T.tocsr(), users, self.regularization)
        self.user_factors, self.item_factors = users, items
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[str]:
        user = self.data.user_to_index.get(int(user_id))
        if user is None or k <= 0:
            return []
        scores = self.item_factors @ self.user_factors[user]
        scores[self.data.confidence.getrow(user).indices] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        take = min(k, len(valid))
        if not take:
            return []
        top = valid[np.argpartition(scores[valid], -take)[-take:]]
        top = top[np.lexsort((self.data.item_ids[top], -scores[top]))]
        return [str(self.data.item_ids[i]) for i in top]

    def objective(self) -> float:
        """Return the full implicit objective without forming the dense preference matrix."""
        # sum_ui (x_u^T y_i)^2 = trace((X^T X)(Y^T Y)); keep this O(f^2)
        # in memory instead of constructing the dense user-item score matrix.
        base = float(np.sum((self.user_factors.T @ self.user_factors) *
                            (self.item_factors.T @ self.item_factors)))
        correction = 0.0
        for user in range(self.data.confidence.shape[0]):
            row = self.data.confidence.getrow(user)
            prediction = self.item_factors[row.indices] @ self.user_factors[user]
            correction += float(np.sum(row.data * (1.0 - prediction) ** 2 - prediction**2))
        penalty = self.regularization * (
            np.sum(self.user_factors**2) + np.sum(self.item_factors**2)
        )
        return base + correction + float(penalty)


def recommend_users(model: object, users: Sequence[int], k: int) -> dict[int, list[str]]:
    return {int(user): model.recommend(int(user), k) for user in users}
