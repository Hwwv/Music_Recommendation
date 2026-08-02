import duckdb
import sys
import numpy as np
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_recommender.data import Interaction
from music_recommender.data_loader import MusicDataLoader



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
        test_frac, valid_frac = splits.metadata['test_fraction'], splits.metadata['validation_fraction']
        total = len(train) + len(valid) + len(test)
        self.assertAlmostEqual(len(test) / total, test_frac, delta=0.05)
        self.assertAlmostEqual(len(valid) / total, valid_frac, delta=0.05)
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
