"""Unit tests for the triangle motif matrix computation."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.motif import triangle_motif_matrix, mix_matrices, count_triangles


class TestMotif(unittest.TestCase):
    def setUp(self):
        # 0 -- 1 -- 2
        # |  / |
        # 3    4          (one triangle: {0,1,3})
        self.A = np.zeros((5, 5), dtype=np.float32)
        for u, v in [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4)]:
            self.A[u, v] = self.A[v, u] = 1.0

    def test_triangle_edges(self):
        B = triangle_motif_matrix(self.A)
        self.assertTrue(np.allclose(B, B.T), "B must be symmetric")
        self.assertTrue(np.allclose(np.diag(B), 0), "B diagonal must be zero")
        for u, v in [(0, 1), (0, 3), (1, 3)]:
            self.assertGreater(B[u, v], 0)
        for u, v in [(1, 2), (1, 4)]:
            self.assertEqual(B[u, v], 0)
        self.assertEqual(count_triangles(self.A), 1)

    def test_no_triangles(self):
        A = np.zeros((4, 4), dtype=np.float32)
        for u, v in [(0, 1), (1, 2), (2, 3)]:
            A[u, v] = A[v, u] = 1.0
        self.assertTrue(np.allclose(triangle_motif_matrix(A), 0))
        self.assertEqual(count_triangles(A), 0)

    def test_complete_graph(self):
        A = np.ones((4, 4), dtype=np.float32)
        np.fill_diagonal(A, 0)
        B = triangle_motif_matrix(A)
        self.assertTrue(np.allclose(B, B.T))
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertGreater(B[i, j], 0)
        self.assertEqual(count_triangles(A), 4)

    def test_mix_matrices(self):
        B = triangle_motif_matrix(self.A)
        self.assertTrue(np.allclose(mix_matrices(self.A, B, 0.0), self.A))
        self.assertTrue(np.allclose(mix_matrices(self.A, B, 1.0), B))
        with self.assertRaises(AssertionError):
            mix_matrices(self.A, B, 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
