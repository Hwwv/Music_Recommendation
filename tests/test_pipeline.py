import sys
import numpy as np
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_recommender.data import Interaction, leave_largest_out_split, normalize_text
from music_recommender.data_loader import MusicDataLoader
from music_recommender.metrics import ndcg_at_k, recall_at_k
from music_recommender.models import ContentRecommender, HybridRecommender, ItemKNN, PopularityRecommender


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            Interaction("a", "i1", 4), Interaction("a", "i2", 2),
            Interaction("b", "i1", 3), Interaction("b", "i3", 5),
            Interaction("c", "i2", 4), Interaction("c", "i3", 2),
        ]
        self.features = {"i1": [1.0, 0.0], "i2": [0.8, 0.2], "i3": [0.0, 1.0]}

    def test_normalization(self):
        self.assertEqual(normalize_text("Beyoncé & JAY-Z"), "beyonce and jay z")

    def test_split_has_no_user_item_leakage(self):
        train, test = leave_largest_out_split(self.rows, minimum_to_split=2, seed=1)
        train_pairs = {(row.user_id, row.item_id) for row in train}
        self.assertTrue(test)
        self.assertTrue(all((user, item) not in train_pairs for user, item in test.items()))

    def test_models_do_not_recommend_seen_items(self):
        pop = PopularityRecommender().fit(self.rows)
        knn = ItemKNN().fit(self.rows)
        content = ContentRecommender().fit(self.rows, self.features)
        hybrid = HybridRecommender(knn, content)
        for model in (pop, knn, content, hybrid):
            self.assertTrue(set(model.recommend("a", 10)).isdisjoint({"i1", "i2"}))

    def test_metrics(self):
        recs = {"a": ["x", "y"], "b": ["z", "x"]}
        truth = {"a": "y", "b": "missing"}
        self.assertEqual(recall_at_k(recs, truth, 2), 0.5)
        self.assertGreater(ndcg_at_k(recs, truth, 2), 0.0)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = MusicDataLoader(
            data_version = "feature_graph_u5_i2_v1", 
            split_version = "feature_split_u5_i2_eval20_seed42_v1", 
            feature_schema_version = "feature_matrix_audio_v1",
        )


    def tearDown(self):
        self.loader.close()


    def test_load_train(self):
        train_interactions = self.loader.load_split_interactions('train')
        self.assertIsNotNone(train_interactions)
        self.assertTrue(len(train_interactions) > 0)
        self.assertTrue(all(isinstance(row, Interaction) for row in train_interactions))
        print(f"Train interactions: {len(train_interactions):,}")


    def test_split_size(self):
        splits = self.loader.load_all_splits()
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
