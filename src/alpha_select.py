"""
alpha_select.py - Stability-guided selection of the motif weight alpha.

Core idea (our contribution):
  For each candidate alpha, we perturb the graph by randomly removing 20% of
  edges (keeping 80%), run the deep NMF model on two independent perturbations,
  and measure how well the two resulting partitions agree using Adjusted Mutual
  Information (AMI). The alpha that produces the most STABLE partitions under
  perturbation is selected -- without looking at any ground-truth labels.

  Based on the literature searched, this approach to automatically selecting
  the motif mixing weight in deep NMF appears insufficiently explored.

Comparison modes:
  - stability_alpha : chosen by partition stability (our method, unsupervised)
  - fixed_alpha     : alpha=0.5 (common default)
  - oracle_alpha    : best alpha chosen using ground-truth labels (upper bound only)
  - no_motif        : alpha=0 (pure adjacency, no motif)
"""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from src.models import DMRALite
from src.evaluate import evaluate_all, compute_modularity


CANDIDATE_ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def subsample_edges(A: np.ndarray, keep_ratio: float = 0.8,
                    rng: np.random.RandomState = None) -> np.ndarray:
    """Randomly remove edges from an adjacency matrix.

    Args:
        A: (n, n) symmetric adjacency matrix.
        keep_ratio: fraction of edges to keep.
        rng: random state for reproducibility.

    Returns:
        A_sub: (n, n) subsampled adjacency matrix (symmetric).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = A.shape[0]
    # Get upper-triangle edge indices
    rows, cols = np.where(np.triu(A > 0, k=1))
    num_edges = len(rows)
    num_keep = int(num_edges * keep_ratio)

    # Randomly select edges to keep
    keep_idx = rng.choice(num_edges, size=num_keep, replace=False)

    A_sub = np.zeros_like(A)
    for idx in keep_idx:
        i, j = rows[idx], cols[idx]
        A_sub[i, j] = A[i, j]
        A_sub[j, i] = A[j, i]

    return A_sub


def stability_score(A: np.ndarray, alpha: float, k: int,
                    n_repeats: int = 5, keep_ratio: float = 0.8,
                    max_iter: int = 200, depth: int = 2,
                    base_seed: int = 42) -> float:
    """Compute the stability score for a given alpha.

    For each repeat:
      1. Create two independent edge-subsampled graphs.
      2. Run DMRA-lite on each.
      3. Compute AMI between the two partitions.

    Returns the mean AMI across repeats.
    """
    ami_scores = []

    for r in range(n_repeats):
        seed_a = base_seed + r * 100
        seed_b = base_seed + r * 100 + 50

        rng_a = np.random.RandomState(seed_a)
        rng_b = np.random.RandomState(seed_b)

        A_sub1 = subsample_edges(A, keep_ratio=keep_ratio, rng=rng_a)
        A_sub2 = subsample_edges(A, keep_ratio=keep_ratio, rng=rng_b)

        # Run model on both subsampled graphs
        model1 = DMRALite(k=k, alpha=alpha, depth=depth,
                          max_iter=max_iter, seed=seed_a)
        model2 = DMRALite(k=k, alpha=alpha, depth=depth,
                          max_iter=max_iter, seed=seed_b)

        labels1 = model1.fit(A_sub1, verbose=False)
        labels2 = model2.fit(A_sub2, verbose=False)

        ami = adjusted_mutual_info_score(labels1, labels2)
        ami_scores.append(ami)

    return float(np.mean(ami_scores))


def select_alpha_stability(A: np.ndarray, k: int,
                           candidates: list = None,
                           n_repeats: int = 5,
                           keep_ratio: float = 0.8,
                           max_iter: int = 200,
                           depth: int = 2,
                           base_seed: int = 42,
                           verbose: bool = True) -> tuple:
    """Select alpha using partition stability (our unsupervised method).

    Args:
        A: adjacency matrix
        k: number of communities
        candidates: list of alpha values to try
        n_repeats: number of perturbation repeats per alpha (R in the paper)
        keep_ratio: fraction of edges to keep in each perturbation
        max_iter: training iterations for each model run
        depth: depth of deep NMF
        base_seed: base random seed
        verbose: print progress

    Returns:
        best_alpha: the alpha with highest stability score
        scores: dict mapping alpha -> stability score
    """
    if candidates is None:
        candidates = CANDIDATE_ALPHAS

    scores = {}
    for alpha in candidates:
        if verbose:
            print(f"  Testing alpha={alpha:.1f} ...", end="", flush=True)
        score = stability_score(
            A, alpha, k,
            n_repeats=n_repeats,
            keep_ratio=keep_ratio,
            max_iter=max_iter,
            depth=depth,
            base_seed=base_seed
        )
        scores[alpha] = score
        if verbose:
            print(f" stability={score:.4f}")

    # Select best alpha (break ties with modularity on the full graph)
    max_score = max(scores.values())
    top_candidates = [a for a, s in scores.items() if abs(s - max_score) < 1e-4]

    if len(top_candidates) == 1:
        best_alpha = top_candidates[0]
    else:
        if verbose:
            print(f"  Tie detected among {top_candidates} (score={max_score:.4f}). Breaking tie with full-graph modularity...")
        best_mod = -float('inf')
        best_alpha = top_candidates[0]
        for a in top_candidates:
            model = DMRALite(k=k, alpha=a, max_iter=max_iter, depth=depth, seed=base_seed)
            pred = model.fit(A, verbose=False)
            mod = compute_modularity(A, pred)
            if verbose:
                print(f"    alpha={a:.1f}: modularity={mod:.4f}")
            if mod > best_mod:
                best_mod = mod
                best_alpha = a

    if verbose:
        print(f"  => Stability-selected alpha = {best_alpha:.1f} "
              f"(score = {scores[best_alpha]:.4f})")

    return best_alpha, scores


def select_alpha_oracle(A: np.ndarray, labels_true: np.ndarray, k: int,
                        candidates: list = None,
                        max_iter: int = 300, depth: int = 2,
                        seed: int = 42, metric: str = 'NMI',
                        n_seeds: int = 3, verbose: bool = True) -> tuple:
    """Select alpha using ground-truth labels (oracle / upper bound).
    THIS IS NOT A REAL METHOD -- it uses labels for evaluation only.
    It represents the best possible alpha if we had perfect knowledge.
    
    If labels_true is None (unlabeled dataset), oracle alpha is undefined
    and returns (None, {}).
    """
    if labels_true is None:
        if verbose:
            print("  Oracle alpha skipped: dataset is unlabeled (labels=None).")
        return None, {}

    if candidates is None:
        candidates = CANDIDATE_ALPHAS

    results = {}
    for alpha in candidates:
        values = []
        for s in range(n_seeds):
            model = DMRALite(k=k, alpha=alpha, depth=depth,
                             max_iter=max_iter, seed=seed + s)
            labels_pred = model.fit(A, verbose=False)
            values.append(evaluate_all(labels_true, labels_pred, A=A)[metric])
        # Average over seeds so the oracle is not decided by one lucky run.
        results[alpha] = float(np.nanmean(values))
        if verbose:
            print(f"  Oracle alpha={alpha:.1f}: {metric}={results[alpha]:.4f}")

    best_alpha = max(results, key=results.get)
    if verbose:
        print(f"  => Oracle-selected alpha = {best_alpha:.1f} "
              f"({metric} = {results[best_alpha]:.4f})")

    return best_alpha, results
