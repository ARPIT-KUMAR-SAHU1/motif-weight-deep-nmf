"""Regression tests for the experiment runner helpers."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.run_all import determine_k, load_datasets
from src.data import load_karate


class TestRunner(unittest.TestCase):
    def test_determine_k_labeled(self):
        A, labels, name = load_karate()
        k, is_labeled = determine_k(A, labels, name)
        self.assertEqual(k, 2)
        self.assertTrue(is_labeled)

    def test_determine_k_unlabeled_does_not_crash(self):
        """The original runner did len(set(None)) and crashed here."""
        A, _, name = load_karate()
        k, is_labeled = determine_k(A, None, name)
        self.assertGreaterEqual(k, 2)
        self.assertFalse(is_labeled)

    def test_quick_dataset_list(self):
        datasets = load_datasets(quick=True)
        self.assertGreaterEqual(len(datasets), 3)
        for A, labels, name in datasets:
            # A real Dolphin graph may legitimately have labels=None; that is
            # handled by determine_k(), not an error.
            self.assertIsNotNone(A)
            self.assertEqual(len(labels) if labels is not None else A.shape[0],
                             A.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
