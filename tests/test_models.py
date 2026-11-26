"""Sanity tests for the community detection models."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_karate
from src.models import DMRALite, SymNMF, compute_laplacian


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A, cls.labels, _ = load_karate()
        cls.k = 2

    def test_symnmf_recovers_karate(self):
        from src.evaluate import evaluate_all
        pred = SymNMF(k=self.k, max_iter=200, seed=42).fit(self.A)
        result = evaluate_all(self.labels, pred)
        self.assertGreaterEqual(result['ACC'], 0.8)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_dmra_lite_recovers_karate(self):
        """Regression test for the rank-1 collapse bug: a model that
        reconstructs the graph but returns random labels must fail here."""
        from src.evaluate import evaluate_all
        pred = DMRALite(k=self.k, alpha=0.5, max_iter=150, seed=42).fit(self.A)
        result = evaluate_all(self.labels, pred)
        self.assertGreaterEqual(result['ACC'], 0.85,
                                f"ACC too low: {result}")
        self.assertGreaterEqual(result['NMI'], 0.5)

    def test_dmra_deterministic(self):
        p1 = DMRALite(k=self.k, alpha=0.5, max_iter=60, seed=7).fit(self.A)
        p2 = DMRALite(k=self.k, alpha=0.5, max_iter=60, seed=7).fit(self.A)
        self.assertTrue(np.array_equal(p1, p2), "same seed must give same labels")

    def test_depths(self):
        for depth in (1, 2, 3):
            pred = DMRALite(k=self.k, alpha=0.5, depth=depth,
                            max_iter=60, seed=0).fit(self.A)
            self.assertEqual(pred.shape, (self.A.shape[0],))
            self.assertLessEqual(pred.min(), 0)
            self.assertLess(pred.max(), self.k)

    def test_laplacian_psd_shape(self):
        L = compute_laplacian(self.A)
        self.assertEqual(L.shape, self.A.shape)
        self.assertTrue(np.all(np.linalg.eigvalsh(L) > -1e-4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
