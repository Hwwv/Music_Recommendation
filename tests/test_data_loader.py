import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import numpy as np

from music_recommender.data_loader import load_experiment_data


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        artifacts = self.root / "artifacts" / "features"
        artifacts.mkdir(parents=True)
        self.artifact = artifacts / "toy_features.parquet"
        self.database = self.root / "integration.duckdb"
        con = duckdb.connect(str(self.database))
        con.execute("CREATE TABLE feature_split_datasets (split_version VARCHAR, dataset_version VARCHAR)")
        con.execute("""
            CREATE TABLE item_feature_schemas (
                feature_schema_version VARCHAR, dataset_version VARCHAR,
                split_version VARCHAR, item_count INTEGER, feature_count INTEGER,
                artifact_path VARCHAR, artifact_sha256 VARCHAR, metadata_json VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE feature_dataset_splits (
                split_version VARCHAR, user_id INTEGER,
                feature_cluster_id VARCHAR, split VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE feature_graph_interactions (
                dataset_version VARCHAR, user_id INTEGER,
                feature_cluster_id VARCHAR, playcount_raw BIGINT
            )
        """)
        con.execute("""
            COPY (
                SELECT * FROM (VALUES
                    ('i3', 3.0, 30.0), ('i1', 1.0, 10.0),
                    ('i4', 4.0, 40.0), ('i2', 2.0, 20.0)
                ) AS x(feature_cluster_id, f0, f1)
            ) TO ? (FORMAT PARQUET)
        """, [str(self.artifact)])
        checksum = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        metadata = json.dumps({"feature_columns": ["f0", "f1"]})
        con.execute("INSERT INTO feature_split_datasets VALUES ('split_v1', 'graph_v1')")
        con.execute(
            "INSERT INTO item_feature_schemas VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["features_v1", "graph_v1", "split_v1", 4, 2,
             "artifacts/features/toy_features.parquet", checksum, metadata],
        )
        con.executemany(
            "INSERT INTO feature_graph_interactions VALUES ('graph_v1', ?, ?, ?)",
            [(10, "i1", 1), (10, "i2", 3), (20, "i3", 2), (20, "i4", 4)],
        )
        con.executemany(
            "INSERT INTO feature_dataset_splits VALUES ('split_v1', ?, ?, ?)",
            [
                (10, "i1", "train"), (10, "i2", "train"),
                (20, "i3", "train"), (20, "i4", "train"),
                (10, "i3", "validation"), (20, "i1", "validation"),
                (10, "i4", "test"), (20, "i2", "test"),
            ],
        )
        con.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_aligned_sparse_and_feature_outputs(self):
        data = load_experiment_data(
            self.database, "split_v1", "features_v1", artifact_root=self.root
        )
        self.assertEqual(data.train_binary.shape, (2, 4))
        self.assertEqual(data.train_binary.nnz, 4)
        self.assertEqual(data.item_features.shape, (4, 2))
        self.assertEqual(data.item_ids.tolist(), ["i1", "i2", "i3", "i4"])
        np.testing.assert_array_equal(data.item_features[:, 0], [1, 2, 3, 4])
        self.assertEqual(data.validation_truth, {0: {2}, 1: {0}})
        self.assertIsNone(data.test_truth)
        self.assertTrue(np.all(data.confidence(0).data == 1))

    def test_test_truth_requires_explicit_unlock(self):
        data = load_experiment_data(
            self.database, "split_v1", "features_v1",
            artifact_root=self.root, allow_test=True
        )
        self.assertEqual(data.test_truth, {0: {3}, 1: {1}})

    def test_negative_confidence_alpha_is_rejected(self):
        data = load_experiment_data(
            self.database, "split_v1", "features_v1", artifact_root=self.root
        )
        with self.assertRaises(ValueError):
            data.confidence(-1)


if __name__ == "__main__":
    unittest.main()
