"""
evaluate.py - Clustering evaluation metrics.

Metrics:
  ACC  : Clustering accuracy with Hungarian matching (scipy.optimize.linear_sum_assignment)
  NMI  : Normalized Mutual Information (sklearn, arithmetic average)
  ARI  : Adjusted Rand Index (sklearn)
  AMI  : Adjusted Mutual Information (sklearn)

NOTE: NMI is known to be biased toward finer partitions when community sizes
are unequal (Jerdee, Kirkley, Newman, Nature Communications 16:11268, 2025).
We report it for comparability with prior work, but also report AMI as a
less biased alternative.
"""
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Clustering accuracy with Hungarian matching.

    Finds the best 1-to-1 mapping from predicted cluster IDs to true labels
    that maximizes accuracy, using the Hungarian algorithm.

    Returns:
        Accuracy in [0, 1].
    """
    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)
    assert len(labels_true) == len(labels_pred)

    n = len(labels_true)
    true_ids = np.unique(labels_true)
    pred_ids = np.unique(labels_pred)

    # Build cost matrix: -count[true_label, pred_label]
    # We use max of the two label sets to handle mismatched counts
    k_true = len(true_ids)
    k_pred = len(pred_ids)
    k = max(k_true, k_pred)

    # Remap to 0..k-1
    true_map = {v: i for i, v in enumerate(true_ids)}
    pred_map = {v: i for i, v in enumerate(pred_ids)}

    cost = np.zeros((k, k), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        ti = true_map[t]
        pi = pred_map[p]
        cost[ti, pi] += 1

    # Hungarian algorithm finds the assignment that maximizes total count
    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = cost[row_ind, col_ind].sum()
    return float(correct) / n


def compute_nmi(labels_true, labels_pred):
    """Normalized Mutual Information (arithmetic average)."""
    return normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')


def compute_ari(labels_true, labels_pred):
    """Adjusted Rand Index."""
    return adjusted_rand_score(labels_true, labels_pred)


def compute_ami(labels_true, labels_pred):
    """Adjusted Mutual Information (less biased than NMI for unequal cluster sizes)."""
    return adjusted_mutual_info_score(labels_true, labels_pred)


def compute_modularity(A: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Newman-Girvan modularity of a partition on adjacency matrix A.
    
    Q = sum_c [ e_c - a_c^2 ]
    where:
      e_c = fraction of graph edges with both endpoints in community c
      a_c = fraction of edge endpoints attached to nodes in community c
    """
    labels = np.asarray(labels_pred, dtype=int)
    m = float(np.sum(A)) / 2.0
    if m == 0:
        return 0.0
    k_vec = np.sum(A, axis=1)
    unique_labels = np.unique(labels)
    Q = 0.0
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sub_A = A[np.ix_(idx, idx)]
        e_c = float(np.sum(sub_A)) / (2.0 * m)
        a_c = float(np.sum(k_vec[idx])) / (2.0 * m)
        Q += e_c - (a_c ** 2)
    return float(Q)


def evaluate_all(labels_true: np.ndarray, labels_pred: np.ndarray, A: np.ndarray = None) -> dict:
    """Compute evaluation metrics.
    
    If labels_true is None (unlabeled dataset):
      Supervised metrics ('ACC', 'NMI', 'ARI', 'AMI') are set to np.nan.
    If A is provided:
      Unsupervised metric 'Modularity' is also computed.

    Returns:
        Dict with keys 'ACC', 'NMI', 'ARI', 'AMI', and optionally 'Modularity'.
    """
    res = {
        'ACC': np.nan,
        'NMI': np.nan,
        'ARI': np.nan,
        'AMI': np.nan,
    }
    if labels_true is not None:
        res['ACC'] = clustering_accuracy(labels_true, labels_pred)
        res['NMI'] = compute_nmi(labels_true, labels_pred)
        res['ARI'] = compute_ari(labels_true, labels_pred)
        res['AMI'] = compute_ami(labels_true, labels_pred)

    if A is not None:
        res['Modularity'] = compute_modularity(A, labels_pred)

    return res
