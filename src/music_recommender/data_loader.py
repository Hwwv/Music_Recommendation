from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from .data import Interaction
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import duckdb


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"


@dataclass(frozen=True)
class Split:
    train: Any
    validation: Any
    test: Any
    metadata: Dict


@dataclass(frozen=True)
class ExperimentData:
    """Model-agnostic inputs shared by CF, CBM, hybrid, and evaluation."""

    train: list[Interaction]
    truth: dict[int, set[str]]
    features: dict[str, list[float]]
    feature_columns: tuple[str, ...]
    feature_metadata: dict[str, Any]
    catalog: tuple[str, ...]
    seen: dict[int, set[str]]
    dataset_version: str
    split_version: str
    feature_schema_version: str
    evaluation_split: str


class MusicDataLoader:
    def __init__(self, data_version: str = "feature_graph_u5_i2_v1",
                 split_version: str = "feature_split_u5_i2_eval20_seed42_v1",
                 feature_schema_version: str = "feature_matrix_audio_v1",
                 data_db_path: Path = None,
                 allow_test: bool = False):
        self.data_version = data_version
        self.split_version = split_version
        self.feature_schema_version = feature_schema_version
        self.data_db_path = data_db_path or INTEGRATION
        self.allow_test = allow_test
        self._con = None
        self._validate_versions()


    def connect(self):
        """Connect to the DuckDB database."""
        if self._con is None:
            self._con = duckdb.connect(str(self.data_db_path), read_only=True)
            self._con.execute("PRAGMA threads=4")
        return self._con


    def close(self):
        """Close the DuckDB connection."""
        if self._con is not None:
            self._con.close()
            self._con = None


    def execute_query(self, query: str, params: list = None, fetch_type: str = "df"):
        """Execute a query against the DuckDB database, return type specified by fetch_type (must be in ['df', 'one', 'all'])."""
        con = duckdb.connect(str(self.data_db_path), read_only=True)
        con.execute("PRAGMA threads=4")
        try:
            if params is None:
                result = con.execute(query)
            else:
                result = con.execute(query, params)
            if fetch_type == "df":
                return result.fetchdf()
            elif fetch_type == "one":
                return result.fetchone()
            elif fetch_type == "all":
                return result.fetchall()
            else:
                raise ValueError(f"Invalid fetch_type {fetch_type}. Must be one of ['df', 'one', 'all'].")
        finally:
            con.close()


    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    def _validate_split_access(self, split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Invalid split input {split}. Split must be 'train', 'validation', or 'test'."
            )
        if split == "test" and not self.allow_test:
            raise ValueError("Loading the test split is not allowed. Set allow_test=True to enable this.")


    def _validate_versions(self):
        """Validate the consistency of the data, split, and feature schema versions."""
        data_version = self.execute_query(f"""
            SELECT dataset_version
            FROM feature_split_datasets
            WHERE split_version = ?
        """, [self.split_version], fetch_type="one")

        if not data_version:
            raise ValueError(f"Data version with split version {self.split_version} not found in the database.")

        split_to_data = data_version[0]
        if split_to_data != self.data_version:
            raise ValueError(f"Split version {self.split_version} is associated with data version {split_to_data}, not {self.data_version}.")

        feature_results = self.execute_query(f"""
            SELECT dataset_version, split_version
            FROM item_feature_schemas
            WHERE feature_schema_version = ?
        """, [self.feature_schema_version], fetch_type="one")
        if not feature_results:
            raise ValueError(f"Data and split versions with feature schema version {self.feature_schema_version} not found in the database.")

        feature_to_data, feature_to_split = feature_results
        if feature_to_data != self.data_version:
            raise ValueError(f"Feature schema version {self.feature_schema_version} is associated with data version {feature_to_data}, not {self.data_version}.")
        if feature_to_split != self.split_version:
            raise ValueError(f"Feature schema version {self.feature_schema_version} is associated with split version {feature_to_split}, not {self.split_version}.")

        print(f"Data version {self.data_version}, split version {self.split_version}, and feature schema version {self.feature_schema_version} are valid and consistent.")



    def load_filtered_feature_graph(self) -> pd.DataFrame:
        """Load the filtered feature graph interactions as a pandas DataFrame."""
        graph_df = self.execute_query(f"""
            SELECT user_id, feature_cluster_id, confidence_log
            FROM feature_graph_interactions
            WHERE dataset_version = ?
        """, [self.data_version], fetch_type="df")
        if graph_df.empty:
            raise ValueError(f"No feature graph interactions found for dataset version {self.data_version}.")

        null_cols = graph_df.columns[graph_df.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"Found null values in columns {null_cols}.")

        graph_df['user_id'] = graph_df['user_id'].astype(int)
        graph_df['feature_cluster_id'] = graph_df['feature_cluster_id'].astype(str)
        graph_df['confidence_log'] = graph_df['confidence_log'].astype(float)
        return graph_df.sort_values(by=['user_id', 'feature_cluster_id']).reset_index(drop=True)


    def load_split(self, split: str) -> pd.DataFrame:
        """Load the specified split (train, validation, or test) as a pandas DataFrame."""
        self._validate_split_access(split)

        split_df = self.execute_query(f"""
            SELECT f.user_id AS user_id, f.feature_cluster_id AS feature_cluster_id, t.playcount_raw AS playcount_raw
            FROM feature_dataset_splits f
            JOIN feature_graph_interactions t ON f.user_id = t.user_id AND f.feature_cluster_id = t.feature_cluster_id AND t.dataset_version = ?
            WHERE f.split_version = ? AND f.split = ?
        """, [self.data_version, self.split_version, split], fetch_type="df")

        return split_df


    def load_all_splits(self) -> Split:
        """Load all splits (train, validation, and test) and metadata as a Split object."""
        if not self.allow_test:
            raise ValueError("Loading all splits is not allowed. Set allow_test=True to enable this.")
        train_df = self.load_split("train")
        validation_df = self.load_split("validation")
        test_df = self.load_split("test")
        metadata = self.execute_query(f"""
            SELECT dataset_version, seed, min_evaluation_items, validation_fraction, test_fraction, created_at
            FROM feature_split_datasets
            WHERE split_version = ?
        """, [self.split_version], fetch_type="one")
        metadata_dict = {
            "dataset_version": metadata[0],
            "seed": metadata[1],
            "min_evaluation_items": metadata[2],
            "validation_fraction": metadata[3],
            "test_fraction": metadata[4],
            "created_at": metadata[5]
        }
        return Split(train=train_df, validation=validation_df, test=test_df, metadata=metadata_dict)


    def load_split_interactions(self, split: str) -> list[Interaction]:
        """Load the specified split as a list of Interaction objects."""
        return self.load_split_records(split)


    def load_split_records(self, split: str) -> list[Interaction]:
        """Load deterministic typed records without importing pandas."""
        self._validate_split_access(split)
        rows = self.execute_query(
            """
            SELECT s.user_id, s.feature_cluster_id, g.playcount_raw
            FROM feature_dataset_splits s
            JOIN feature_graph_interactions g
              ON s.user_id = g.user_id
             AND s.feature_cluster_id = g.feature_cluster_id
             AND g.dataset_version = ?
            WHERE s.split_version = ? AND s.split = ?
            ORDER BY s.user_id, s.feature_cluster_id
            """,
            [self.data_version, self.split_version, split],
            fetch_type="all",
        )
        return [
            Interaction(int(user), str(item), float(playcount))
            for user, item, playcount in rows
        ]


    def load_truth_records(self, split: str) -> dict[int, set[str]]:
        """Load held-out item sets with stable Python ID types."""
        self._validate_split_access(split)
        rows = self.execute_query(
            """
            SELECT user_id, feature_cluster_id
            FROM feature_dataset_splits
            WHERE split_version = ? AND split = ?
            ORDER BY user_id, feature_cluster_id
            """,
            [self.split_version, split],
            fetch_type="all",
        )
        truth: dict[int, set[str]] = defaultdict(set)
        for user, item in rows:
            truth[int(user)].add(str(item))
        return dict(truth)


    def _feature_registry(self) -> tuple[Path, dict[str, Any]]:
        result = self.execute_query(
            """
            SELECT artifact_path, artifact_sha256, metadata_json
            FROM item_feature_schemas
            WHERE feature_schema_version = ?
              AND dataset_version = ?
              AND split_version = ?
            """,
            [self.feature_schema_version, self.data_version, self.split_version],
            fetch_type="one",
        )
        if not result:
            raise ValueError(
                f"No feature artifact registered for {self.feature_schema_version}, "
                f"{self.data_version}, and {self.split_version}."
            )
        artifact = ROOT / result[0]
        if not artifact.exists():
            raise FileNotFoundError(f"Feature matrix file not found at {artifact}.")
        if self._file_sha256(artifact) != result[1]:
            raise ValueError(f"Feature matrix checksum does not match registry: {artifact}")
        return artifact, json.loads(result[2])


    def load_feature_records(self) -> dict[str, list[float]]:
        """Load the registered feature artifact without requiring pandas."""
        artifact, metadata = self._feature_registry()
        expected_columns = list(metadata.get("feature_columns", []))
        con = duckdb.connect(str(self.data_db_path), read_only=True)
        try:
            result = con.execute(
                "SELECT * FROM read_parquet(?) ORDER BY feature_cluster_id",
                [str(artifact)],
            )
            actual_columns = [column[0] for column in result.description]
            rows = result.fetchall()
        finally:
            con.close()
        registered_columns = ["feature_cluster_id", *expected_columns]
        if actual_columns != registered_columns:
            raise ValueError(
                "Feature artifact columns do not match registered metadata: "
                f"expected {registered_columns!r}, "
                f"got {actual_columns!r}"
            )
        features = {
            str(row[0]): [float(value) for value in row[1:]]
            for row in rows
        }
        if not features:
            raise ValueError(f"Feature matrix at {artifact} is empty.")
        expected_items = int(metadata.get("item_count", len(features)))
        expected_width = int(metadata.get("feature_count", len(expected_columns)))
        if len(features) != expected_items or len(expected_columns) != expected_width:
            raise ValueError(
                "Feature artifact dimensions do not match registered metadata: "
                f"rows={len(features)}/{expected_items}, "
                f"columns={len(expected_columns)}/{expected_width}"
            )
        return features


    def load_experiment(self, evaluation_split: str = "validation") -> ExperimentData:
        """Load and cross-check the complete shared experiment contract."""
        if evaluation_split == "train":
            raise ValueError("evaluation_split must be 'validation' or 'test'")
        train = self.load_split_records("train")
        truth = self.load_truth_records(evaluation_split)
        features = self.load_feature_records()
        _, feature_metadata = self._feature_registry()
        feature_columns = tuple(feature_metadata["feature_columns"])
        catalog = tuple(sorted(features))
        train_items = {row.item_id for row in train}
        feature_items = set(catalog)
        if train_items != feature_items:
            missing_features = sorted(train_items - feature_items)[:5]
            extra_features = sorted(feature_items - train_items)[:5]
            raise ValueError(
                "Training and feature catalogs differ; "
                f"missing features={missing_features}, extra features={extra_features}"
            )
        truth_items = set().union(*truth.values()) if truth else set()
        if not truth_items <= feature_items:
            raise ValueError(
                f"Evaluation truth contains items outside the feature catalog: "
                f"{sorted(truth_items - feature_items)[:5]}"
            )
        seen_accumulator: dict[int, set[str]] = defaultdict(set)
        for row in train:
            if row.item_id in seen_accumulator[row.user_id]:
                raise ValueError(
                    f"Duplicate training user-item pair: {(row.user_id, row.item_id)!r}"
                )
            seen_accumulator[row.user_id].add(row.item_id)
        return ExperimentData(
            train=train,
            truth=truth,
            features=features,
            feature_columns=feature_columns,
            feature_metadata=feature_metadata,
            catalog=catalog,
            seen=dict(seen_accumulator),
            dataset_version=self.data_version,
            split_version=self.split_version,
            feature_schema_version=self.feature_schema_version,
            evaluation_split=evaluation_split,
        )


    def load_feature_matrix(self) -> pd.DataFrame:
        """Load the feature matrix as a pandas DataFrame."""
        result = self.execute_query(f"""
            SELECT artifact_path
            FROM item_feature_schemas
            WHERE feature_schema_version = ? AND dataset_version = ? AND split_version = ?
        """, [self.feature_schema_version, self.data_version, self.split_version], fetch_type="one")
        if not result or not result[0]:
            raise ValueError(f"No feature matrix found for schema version {self.feature_schema_version}, dataset version {self.data_version}, and split version {self.split_version}.")
        path = result[0]
        full_path = ROOT / path
        if not full_path.exists():
            raise FileNotFoundError(f"Feature matrix file not found at {full_path}.")
        import pandas as pd

        df = pd.read_parquet(full_path)
        if df.empty:
            raise ValueError(f"Feature matrix at {full_path} is empty.")
        return df


    def load_feature_mappings(self) -> dict[str, list[float]]:
        """Load the feature matrix as a dictionary of item_id to feature vector."""
        return self.load_feature_records()


    def load_split_truth(self, split: str) -> dict[int, set[str]]:
        """Load the specified split as a dictionary of user_id to list of item_ids."""
        return self.load_truth_records(split)


    def load_single_split_truth(self, split: str) -> dict[int, str]:
        """Load the specified split as a dictionary of user_id to single item_id for one single item."""
        truth = self.load_truth_records(split)
        invalid_count = sum(len(items) != 1 for items in truth.values())
        if invalid_count:
            raise ValueError(
                f"Found {invalid_count} users without exactly one item in the {split} split. "
                "This function expects one item per user."
            )
        return {user: next(iter(items)) for user, items in truth.items()}
