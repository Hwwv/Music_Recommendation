from collections import defaultdict
from dataclasses import dataclass
from .data import Interaction
from pathlib import Path
from typing import Dict
import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict


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
        if not self.allow_test and split == "test":
            raise ValueError("Loading the test split is not allowed. Set allow_test=True to enable this.")
        
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split input {split}. Split must be 'train', 'validation', or 'test'.")
        
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
        result = []
        split_df = self.load_split(split)
        for _, row in split_df.iterrows():
            user = row['user_id']
            item = row['feature_cluster_id']
            play_count = float(row['playcount_raw'])
            result.append(Interaction(user_id=user, item_id=item, play_count=play_count))
        return result


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
        df = pd.read_parquet(full_path)
        if df.empty:
            raise ValueError(f"Feature matrix at {full_path} is empty.")
        return df


    def load_feature_mappings(self) -> dict[str, list[float]]:
        """Load the feature matrix as a dictionary of item_id to feature vector."""
        feature_matrix = self.load_feature_matrix()
        return {row['feature_cluster_id']: [float(row[feature]) for feature in feature_matrix.columns if feature != 'feature_cluster_id'] for _, row in feature_matrix.iterrows()}

    
    def load_split_truth(self, split: str) -> dict[int, set[str]]:
        """Load the specified split as a dictionary of user_id to list of item_ids."""
        split_df = self.load_split(split)
        truths: dict[int, set[str]] = defaultdict(set)
        for _, row in split_df.iterrows():
            truths[row['user_id']].add(row['feature_cluster_id'])
        return dict(truths)


    def load_single_split_truth(self, split: str) -> dict[int, str]:
        """Load the specified split as a dictionary of user_id to single item_id for one single item."""
        split_df = self.load_split(split)

        item_counts = split_df.groupby('user_id')['feature_cluster_id'].count()
        invalid_count = item_counts[item_counts != 1].count()
        if invalid_count > 0:
            raise ValueError(f"Found {invalid_count} users with more than one item in the {split} split. This function expects only one item per user.")

        truths = dict(zip(split_df['user_id'].astype(int), split_df['feature_cluster_id'].astype(str)))
        return truths