"""Tests for clustering evaluation metrics."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import (
    clustering_accuracy, compute_nmi, compute_ari, compute_ami,
    compute_modularity, evaluate_all,
)


class TestEvaluate(unittest.TestCase):
    def test_hungarian_accuracy_with_permuted_labels(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([2, 2, 0, 0, 1, 1])  # perfect, but renamed
        self.assertAlmostEqual(clustering_accuracy(truth, pred), 1.0)

    def test_hungarian_accuracy_with_extra_pred_cluster(self):
        truth = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 2])  # one singleton split off
        self.assertAlmostEqual(clustering_accuracy(truth, pred), 0.75)

    def test_perfect_scores(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(compute_nmi(y, y), 1.0)
        self.assertAlmostEqual(compute_ari(y, y), 1.0)
        self.assertAlmostEqual(compute_ami(y, y), 1.0)

    def test_modularity(self):
        # Two disjoint cliques must have positive modularity.
        A = np.zeros((6, 6), dtype=np.float32)
        for block in (range(3), range(3, 6)):
            for i in block:
                for j in block:
                    if i != j:
                        A[i, j] = 1.0
        labels = np.array([0, 0, 0, 1, 1, 1])
        self.assertGreater(compute_modularity(A, labels), 0.3)

    def test_evaluate_all_unlabeled(self):
        res = evaluate_all(None, np.array([0, 1, 0, 1]), A=np.eye(4))
        self.assertTrue(np.isnan(res['ACC']))
        self.assertIn('Modularity', res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
