"""Tests for offline-first dataset loading."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import data


class TestData(unittest.TestCase):
    def test_karate(self):
        A, y, name = data.load_karate()
        self.assertEqual(A.shape, (34, 34))
        self.assertTrue(np.allclose(A, A.T))
        self.assertEqual(len(set(y)), 2)
        self.assertEqual(name, "Karate")

    def test_offline_fallbacks_are_well_formed(self):
        old, old_cache = data.FORCE_OFFLINE, data._CACHE_DIR
        data.FORCE_OFFLINE = True
        data._CACHE_DIR = os.path.join(tempfile.gettempdir(), "sgdmnmf_empty_cache")
        os.makedirs(data._CACHE_DIR, exist_ok=True)
        try:
            for loader in (data.load_dolphin, data.load_email,
                           data.load_cora, data.load_citeseer, data.load_lfr):
                A, y, name = loader()
                self.assertIsNotNone(A, name)
                self.assertEqual(A.shape[0], A.shape[1])
                self.assertTrue(np.allclose(A, A.T), name)
                self.assertTrue(np.allclose(np.diag(A), 0), name)
                self.assertIsNotNone(y, name)
                self.assertEqual(len(y), A.shape[0])
                self.assertTrue(data.is_synthetic(name), name)
        finally:
            data.FORCE_OFFLINE = old
            data._CACHE_DIR = old_cache

    def test_load_all_datasets_never_raises(self):
        old, old_cache = data.FORCE_OFFLINE, data._CACHE_DIR
        data.FORCE_OFFLINE = True
        data._CACHE_DIR = os.path.join(tempfile.gettempdir(), "sgdmnmf_empty_cache2")
        os.makedirs(data._CACHE_DIR, exist_ok=True)
        try:
            datasets = data.load_all_datasets(include_large=False, lfr_mus=(0.3,))
            self.assertGreaterEqual(len(datasets), 3)
            for A, y, name in datasets:
                self.assertEqual(A.shape[0], A.shape[1])
        finally:
            data.FORCE_OFFLINE = old
            data._CACHE_DIR = old_cache


if __name__ == "__main__":
    unittest.main(verbosity=2)
