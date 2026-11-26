"""Tests for stability-guided alpha selection."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.alpha_select import subsample_edges, stability_score, CANDIDATE_ALPHAS
from src.data import load_karate


class TestAlphaSelect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A, _, _ = load_karate()

    def test_subsample_edges(self):
        sub = subsample_edges(self.A, keep_ratio=0.8,
                              rng=np.random.RandomState(0))
        self.assertTrue(np.allclose(sub, sub.T))
        self.assertEqual(int(sub.sum()), int(0.8 * int(self.A.sum())))
        self.assertTrue(np.all(sub <= self.A))

    def test_stability_deterministic(self):
        s1 = stability_score(self.A, 0.5, k=2, n_repeats=1,
                             max_iter=20, keep_ratio=0.8)
        s2 = stability_score(self.A, 0.5, k=2, n_repeats=1,
                             max_iter=20, keep_ratio=0.8)
        self.assertAlmostEqual(s1, s2)

    def test_stability_range(self):
        s = stability_score(self.A, 0.0, k=2, n_repeats=1, max_iter=20)
        self.assertGreaterEqual(s, -1.0)
        self.assertLessEqual(s, 1.0)
        self.assertIn(0.5, CANDIDATE_ALPHAS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
