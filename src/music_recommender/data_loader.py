"""Versioned, leakage-safe loading for full recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ExperimentData:
    dataset_version: str
    split_version: str
    feature_schema_version: str
    train_binary: sparse.csr_matrix
    train_log_playcount: sparse.csr_matrix
    item_features: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_to_index: dict[int, int]
    item_to_index: dict[str, int]
    feature_names: tuple[str, ...]
    validation_truth: dict[int, set[int]]
    test_truth: dict[int, set[int]] | None
    evaluation_users: np.ndarray

    def confidence(self, alpha: float) -> sparse.csr_matrix:
        """Return 1 + alpha * log(1 + playcount) on observed train entries."""
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        matrix = self.train_log_playcount.copy()
        matrix.data = (1.0 + alpha * matrix.data).astype(np.float32, copy=False)
        return matrix


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth_from_arrays(users: np.ndarray, items: np.ndarray) -> dict[int, set[int]]:
    truth: dict[int, set[int]] = {}
    for user, item in zip(users, items, strict=True):
        truth.setdefault(int(user), set()).add(int(item))
    return truth


def load_experiment_data(
    database_path: str | Path,
    split_version: str,
    feature_schema_version: str,
    *,
    artifact_root: str | Path | None = None,
    allow_test: bool = False,
    verify_checksum: bool = True,
) -> ExperimentData:
    """Load aligned sparse interactions, item features, and held-out truth.

    User and item indices are deterministic: original user IDs and feature
    cluster IDs are sorted ascending. Test truth is omitted unless explicitly
    unlocked with ``allow_test=True``.
    """
    database_path = Path(database_path).resolve()
    artifact_root = (
        Path(artifact_root).resolve() if artifact_root is not None else database_path.parents[2]
    )
    con = duckdb.connect(str(database_path), read_only=True)

    split_row = con.execute(
        """
        SELECT dataset_version
        FROM feature_split_datasets
        WHERE split_version = ?
        """,
        [split_version],
    ).fetchone()
    if split_row is None:
        con.close()
        raise ValueError(f"unknown split version: {split_version}")
    dataset_version = str(split_row[0])

    schema_row = con.execute(
        """
        SELECT dataset_version, split_version, item_count, feature_count,
               artifact_path, artifact_sha256, metadata_json
        FROM item_feature_schemas
        WHERE feature_schema_version = ?
        """,
        [feature_schema_version],
    ).fetchone()
    if schema_row is None:
        con.close()
        raise ValueError(f"unknown feature schema version: {feature_schema_version}")
    if schema_row[0] != dataset_version or schema_row[1] != split_version:
        con.close()
        raise ValueError(
            f"feature schema {feature_schema_version!r} is registered for "
            f"{schema_row[0:2]}, not {(dataset_version, split_version)}"
        )

    expected_items, expected_features = int(schema_row[2]), int(schema_row[3])
    artifact_path = Path(schema_row[4])
    if not artifact_path.is_absolute():
        artifact_path = artifact_root / artifact_path
    if not artifact_path.exists():
        con.close()
        raise FileNotFoundError(f"feature artifact not found: {artifact_path}")
    if verify_checksum and _sha256(artifact_path) != schema_row[5]:
        con.close()
        raise ValueError(f"feature artifact checksum mismatch: {artifact_path}")
    metadata = json.loads(schema_row[6])
    feature_names = tuple(metadata["feature_columns"])
    if len(feature_names) != expected_features:
        con.close()
        raise ValueError("registered feature count does not match metadata")

    parquet = _sql_string(str(artifact_path))
    item_result = con.execute(
        f"""
        SELECT feature_cluster_id, {', '.join(feature_names)}
        FROM read_parquet({parquet})
        ORDER BY feature_cluster_id
        """
    ).fetchnumpy()
    item_ids = np.asarray(item_result.pop("feature_cluster_id"), dtype=str)
    item_features = np.column_stack(
        [np.asarray(item_result[name], dtype=np.float32) for name in feature_names]
    )
    if item_features.shape != (expected_items, expected_features):
        con.close()
        raise ValueError(
            f"feature matrix shape {item_features.shape} does not match "
            f"registered shape {(expected_items, expected_features)}"
        )
    if len(np.unique(item_ids)) != len(item_ids) or not np.isfinite(item_features).all():
        con.close()
        raise ValueError("feature matrix contains duplicate IDs or non-finite values")

    user_ids = np.asarray(
        con.execute(
            """
            SELECT DISTINCT user_id FROM feature_dataset_splits
            WHERE split_version = ? AND split = 'train'
            ORDER BY user_id
            """,
            [split_version],
        ).fetchnumpy()["user_id"],
        dtype=np.int64,
    )
    user_to_index = {int(user): index for index, user in enumerate(user_ids)}
    item_to_index = {str(item): index for index, item in enumerate(item_ids)}

    index_ctes = f"""
        WITH user_map AS (
            SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS user_index
            FROM (SELECT DISTINCT user_id FROM feature_dataset_splits
                  WHERE split_version = {_sql_string(split_version)} AND split = 'train')
        ), item_map AS (
            SELECT feature_cluster_id,
                   row_number() OVER (ORDER BY feature_cluster_id) - 1 AS item_index
            FROM read_parquet({parquet})
        )
    """
    train = con.execute(
        index_ctes
        + f"""
        SELECT u.user_index, m.item_index, g.playcount_raw
        FROM feature_dataset_splits s
        JOIN feature_split_datasets d USING (split_version)
        JOIN feature_graph_interactions g
          ON g.dataset_version = d.dataset_version
         AND g.user_id = s.user_id
         AND g.feature_cluster_id = s.feature_cluster_id
        JOIN user_map u ON s.user_id = u.user_id
        JOIN item_map m ON s.feature_cluster_id = m.feature_cluster_id
        WHERE s.split_version = {_sql_string(split_version)} AND s.split = 'train'
        """
    ).fetchnumpy()
    rows = np.asarray(train["user_index"], dtype=np.int32)
    columns = np.asarray(train["item_index"], dtype=np.int32)
    playcounts = np.asarray(train["playcount_raw"], dtype=np.float32)
    shape = (len(user_ids), len(item_ids))
    train_binary = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=shape
    )
    train_log_playcount = sparse.csr_matrix(
        (np.log1p(playcounts).astype(np.float32), (rows, columns)), shape=shape
    )
    train_binary.sort_indices()
    train_log_playcount.sort_indices()
    if train_binary.nnz != len(rows) or train_binary.nnz != train_log_playcount.nnz:
        con.close()
        raise ValueError("duplicate or misaligned train interactions")

    def load_truth(split: str) -> dict[int, set[int]]:
        arrays = con.execute(
            index_ctes
            + f"""
            SELECT u.user_index, m.item_index
            FROM feature_dataset_splits s
            JOIN user_map u ON s.user_id = u.user_id
            JOIN item_map m ON s.feature_cluster_id = m.feature_cluster_id
            WHERE s.split_version = {_sql_string(split_version)}
              AND s.split = {_sql_string(split)}
            ORDER BY u.user_index, m.item_index
            """
        ).fetchnumpy()
        return _truth_from_arrays(arrays["user_index"], arrays["item_index"])

    validation_truth = load_truth("validation")
    test_truth = load_truth("test") if allow_test else None
    evaluation_users = np.asarray(sorted(validation_truth), dtype=np.int32)
    con.close()

    # A user's held-out items must never appear in that user's training row.
    for user, held_out in validation_truth.items():
        start, end = train_binary.indptr[user : user + 2]
        if held_out.intersection(map(int, train_binary.indices[start:end])):
            raise ValueError(f"validation leakage for user index {user}")
    if test_truth is not None:
        for user, held_out in test_truth.items():
            start, end = train_binary.indptr[user : user + 2]
            if held_out.intersection(map(int, train_binary.indices[start:end])):
                raise ValueError(f"test leakage for user index {user}")

    return ExperimentData(
        dataset_version=dataset_version,
        split_version=split_version,
        feature_schema_version=feature_schema_version,
        train_binary=train_binary,
        train_log_playcount=train_log_playcount,
        item_features=item_features,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        feature_names=feature_names,
        validation_truth=validation_truth,
        test_truth=test_truth,
        evaluation_users=evaluation_users,
    )
