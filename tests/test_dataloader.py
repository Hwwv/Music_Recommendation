import duckdb
import sys
import numpy as np
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_recommender.data import Interaction
from music_recommender.data_loader import ExperimentData, MusicDataLoader



class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = MusicDataLoader(
            data_version = "feature_graph_u5_i2_v1", 
            split_version = "feature_split_u5_i2_eval20_seed42_v1", 
            feature_schema_version = "feature_matrix_audio_v1",
        )


    def tearDown(self):
        self.loader.close()


    def test_execute(self):
        result = self.loader.execute_query("SELECT * FROM item_feature_schemas", fetch_type="df")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn("dataset_version", result.columns)
        self.assertIn("split_version", result.columns)
        self.assertIn("feature_schema_version", result.columns)


    def test_execute_invalid_query(self):
        with self.assertRaises(duckdb.CatalogException):
            self.loader.execute_query("SELECT * FROM non_existent_table", fetch_type="df")


    def test_execute_invalid_fetch_type(self):
        with self.assertRaises(ValueError):
            self.loader.execute_query("SELECT * FROM feature_split_datasets", fetch_type="invalid")


    def test_load_train(self):
        train_interactions = self.loader.load_split_interactions('train')
        self.assertIsNotNone(train_interactions)
        self.assertTrue(len(train_interactions) > 0)
        self.assertTrue(all(isinstance(row, Interaction) for row in train_interactions))
        print(f"Train interactions: {len(train_interactions):,}")


    def test_pandas_free_record_contract(self):
        records = self.loader.load_split_records("validation")
        self.assertTrue(records)
        self.assertTrue(all(isinstance(row, Interaction) for row in records))
        self.assertEqual(
            records,
            sorted(records, key=lambda row: (row.user_id, row.item_id)),
        )


    def test_experiment_contract_is_aligned(self):
        experiment = self.loader.load_experiment("validation")
        self.assertIsInstance(experiment, ExperimentData)
        train_items = {row.item_id for row in experiment.train}
        truth_items = set().union(*experiment.truth.values())
        self.assertEqual(train_items, set(experiment.catalog))
        self.assertEqual(set(experiment.features), set(experiment.catalog))
        self.assertEqual(len(experiment.feature_columns), 29)
        self.assertEqual(experiment.feature_metadata["feature_count"], 29)
        self.assertTrue(truth_items <= set(experiment.catalog))
        expected_seen = defaultdict(set)
        for row in experiment.train:
            expected_seen[row.user_id].add(row.item_id)
        self.assertEqual(experiment.seen, dict(expected_seen))

    
    def test_load_test_not_allowed(self):
        self.loader.allow_test = False
        with self.assertRaises(ValueError):
            self.loader.load_split('test')


    def test_split_size(self):
        self.loader.allow_test = True
        splits = self.loader.load_all_splits()
        self.assertIsNotNone(splits)
        self.assertTrue(hasattr(splits, 'train'))
        self.assertTrue(hasattr(splits, 'validation'))
        self.assertTrue(hasattr(splits, 'test'))
        self.assertTrue(hasattr(splits, 'metadata'))
        self.assertTrue(len(splits.train) > 0)
        train, valid, test = splits.train, splits.validation, splits.test
        test_frac = splits.metadata['test_fraction']
        valid_frac = splits.metadata['validation_fraction']
        minimum = splits.metadata['min_evaluation_items']
        user_counts = self.loader.execute_query(
            """
            SELECT user_id, count(*) AS total,
                   count(*) FILTER (WHERE split = 'validation') AS validation_count,
                   count(*) FILTER (WHERE split = 'test') AS test_count
            FROM feature_dataset_splits
            WHERE split_version = ?
            GROUP BY user_id
            """,
            [self.loader.split_version],
            fetch_type="all",
        )
        for _, user_total, validation_count, test_count in user_counts:
            if user_total >= minimum:
                self.assertEqual(validation_count, max(1, int(user_total * valid_frac)))
                self.assertEqual(test_count, max(1, int(user_total * test_frac)))
            else:
                self.assertEqual(validation_count, 0)
                self.assertEqual(test_count, 0)
        total = len(train) + len(valid) + len(test)
        print(f"Train: {len(train):,} ({len(train)/total:.2%}), ")
        print(f"Validation: {len(valid):,} ({len(valid)/total:.2%}), ")
        print(f"Test: {len(test):,} ({len(test)/total:.2%}), ")
        print(f"Total: {total:,}")
  

    def test_feature_mapping(self):
        feature_map = self.loader.load_feature_mappings()
        self.assertIsInstance(feature_map, dict)
        self.assertTrue(len(feature_map) > 0)
        for item_id, features in feature_map.items():
            self.assertTrue(isinstance(item_id, str))
            self.assertTrue(isinstance(features, list))
            self.assertTrue(len(features) > 0)
            self.assertTrue(all(isinstance(f, (float, int, np.float64)) for f in features))
        l = len(next(iter(feature_map.values())))
        self.assertTrue(all(len(v) == l for v in feature_map.values()))
        print(f"{len(feature_map):,} feature maps with dimension {l}.")


    def test_load_split_truth(self):
        self.loader.allow_test = True
        truths = self.loader.load_split_truth('test')
        self.assertIsInstance(truths, dict)
        self.assertTrue(len(truths) > 0)
        for user_id, item_ids in truths.items():
            self.assertTrue(isinstance(user_id, int))
            self.assertTrue(isinstance(item_ids, set))
            self.assertTrue(len(item_ids) > 0)
            self.assertTrue(all(isinstance(i, str) for i in item_ids))
        print(f"{len(truths):,} users with truth items in the test split.")


if __name__ == "__main__":
    unittest.main()
