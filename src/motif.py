"""
motif.py - Triangle motif matrix computation.

The triangle motif matrix B captures higher-order structure:
  B[i,j] = number of triangles that edge (i,j) participates in.

Formula: B = A ∘ (A @ A)   (Hadamard product of A with A-squared)
  - (A @ A)[i,j] counts 2-hop paths from i to j
  - Multiplying element-wise by A keeps only those paths that also
    have a direct edge, i.e., triangles.

We zero the diagonal and rescale B so max(B) = max(A).
"""
import numpy as np


def triangle_motif_matrix(A: np.ndarray) -> np.ndarray:
    """Compute the triangle motif matrix from adjacency matrix A.

    Args:
        A: (n, n) symmetric binary adjacency matrix (float32).

    Returns:
        B: (n, n) triangle motif matrix, rescaled so max(B) == max(A).
           If the graph has no triangles, returns a zero matrix.
    """
    assert A.shape[0] == A.shape[1], "A must be square"

    # A @ A counts 2-hop paths; Hadamard with A keeps only closed triangles
    A2 = A @ A
    B = A * A2  # element-wise (Hadamard) product

    # Zero diagonal (self-connections are not meaningful)
    np.fill_diagonal(B, 0)

    # Rescale so max(B) == max(A) -- this keeps the two matrices
    # on comparable scales when we mix them.
    max_B = B.max()
    max_A = A.max()
    if max_B > 0 and max_A > 0:
        B = B * (max_A / max_B)

    return B.astype(np.float32)


def mix_matrices(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Mix adjacency and motif matrices: M = (1 - alpha) * A + alpha * B.

    Args:
        A: adjacency matrix (n, n)
        B: motif matrix (n, n), should be pre-scaled
        alpha: mixing weight in [0, 1]. alpha=0 means no motif, alpha=1 means motif only.

    Returns:
        M: mixed matrix (n, n)
    """
    assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
    return ((1.0 - alpha) * A + alpha * B).astype(np.float32)


def count_triangles(A: np.ndarray) -> int:
    """Count total number of triangles in an undirected graph.
    Each triangle is counted once (not 3 or 6 times)."""
    # tr(A^3) / 6 gives the number of triangles in an undirected graph.
    A3 = A @ A @ A
    return int(np.trace(A3)) // 6
