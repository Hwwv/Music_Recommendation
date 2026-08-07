import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_recommender.cf import ImplicitALS, SparseItemKNN, build_interaction_matrix
from music_recommender.data import Interaction


class CollaborativeFilteringTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            Interaction(1, "a", 4), Interaction(1, "b", 1),
            Interaction(2, "a", 2), Interaction(2, "c", 3),
            Interaction(3, "b", 5), Interaction(3, "d", 1),
            Interaction(4, "c", 2), Interaction(4, "d", 4),
        ]

    def test_confidence_contract(self):
        data = build_interaction_matrix(self.rows, alpha=2)
        value = data.confidence[data.user_to_index[1], data.item_to_index["a"]]
        self.assertAlmostEqual(value, 1 + 2 * np.log1p(4))
        self.assertEqual(data.confidence.nnz, len(self.rows))

    def test_duplicate_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            build_interaction_matrix(self.rows + [Interaction(1, "a", 1)])

    def test_item_knn_cosine_and_bm25_exclude_seen(self):
        for weighting in ("cosine", "bm25"):
            model = SparseItemKNN(
                neighbours=3, weighting=weighting, min_cooccurrence=1, block_size=2
            ).fit(self.rows)
            self.assertTrue(set(model.recommend(1, 4)).isdisjoint({"a", "b"}))
            self.assertLessEqual(max(np.diff(model.similarity.indptr)), 3)

    def test_minimum_cooccurrence_filters_edges(self):
        model = SparseItemKNN(neighbours=3, min_cooccurrence=2, block_size=2).fit(self.rows)
        self.assertEqual(model.similarity.nnz, 0)

    def test_als_decreases_objective_and_excludes_seen(self):
        one = ImplicitALS(factors=3, iterations=1, regularization=0.1, seed=7).fit(self.rows)
        three = ImplicitALS(factors=3, iterations=3, regularization=0.1, seed=7).fit(self.rows)
        self.assertLessEqual(three.objective(), one.objective() + 1e-8)
        self.assertTrue(set(three.recommend(1, 4)).isdisjoint({"a", "b"}))

    def test_score_not_processed_type(self):
        cf_model = SparseItemKNN(neighbours=3, min_cooccurrence=2, block_size=2).fit(self.rows)
        scores = cf_model.score_not_processed(1)
        self.assertEqual(type(scores), np.ndarray)

    def test_score_not_processed(self):
        cf_model = SparseItemKNN(neighbours=3, min_cooccurrence=2, block_size=2).fit(self.rows)
        scores = cf_model.score_not_processed(2)
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
