import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_recommender.data import Interaction, leave_largest_out_split, normalize_text
from music_recommender.metrics import ndcg_at_k, recall_at_k
from music_recommender.evaluation import assert_unseen_recommendations, evaluate_topk
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

    def test_multi_item_topk_evaluation(self):
        recs = {"a": ["i2", "i3", "x"], "b": ["i1", "z", "y"]}
        truth = {"a": {"i1", "i2"}, "b": {"z"}}
        result = evaluate_topk(recs, truth, {"i1", "i2", "i3", "x", "y", "z"}, [2])
        self.assertAlmostEqual(result["recall@2"], 0.75)
        self.assertEqual(result["hit_rate@2"], 1.0)
        self.assertGreater(result["ndcg@2"], 0.0)

        limited = evaluate_topk(
            {"a": ["i1", "i2"]}, {"a": {"i1", "i2", "i3", "i4"}},
            {"i1", "i2", "i3", "i4"}, [2]
        )
        self.assertEqual(limited["recall@2"], 0.5)

    def test_seen_item_guard(self):
        with self.assertRaises(AssertionError):
            assert_unseen_recommendations({"a": ["seen"]}, {"a": {"seen"}})


if __name__ == "__main__":
    unittest.main()
