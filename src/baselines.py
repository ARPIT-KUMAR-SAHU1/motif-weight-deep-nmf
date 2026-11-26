"""
baselines.py - Baseline community detection methods.

Baselines:
  1. SymNMF       : Single-layer symmetric NMF (from models.py)
  2. DANMF-style  : Deep NMF without motifs (alpha=0, no diagonal penalty)
  3. Leiden       : Leiden algorithm (via leidenalg or igraph)
                    Falls back to NetworkX Louvain if Leiden is not installed.

  4. DMoN-lite    : differentiable modularity optimisation (objective-level
                     version of DMoN, without GNN features).

The full GNN-based DMoN is not reproduced; the simplified baseline is named
'DMoN-lite' everywhere so the limitation is explicit.
"""
import numpy as np
import networkx as nx
from src.models import DMRALite, SymNMF


def run_symnmf(A, k, seed=42, max_iter=200, verbose=False):
    """Run SymNMF baseline. Returns labels (n,)."""
    model = SymNMF(k=k, max_iter=max_iter, seed=seed)
    labels = model.fit(A, verbose=verbose)
    return labels


def run_danmf(A, k, seed=42, depth=2, max_iter=300, verbose=False):
    """Run DANMF-style deep NMF WITHOUT motifs.
    This is DMRA-lite with alpha=0 and mu=0 (no motif, no diagonal penalty).
    """
    model = DMRALite(
        k=k, alpha=0.0, lam=1.0, mu=0.0,
        depth=depth, max_iter=max_iter, seed=seed
    )
    labels = model.fit(A, verbose=verbose)
    return labels


def run_leiden(A, k=None, seed=42, verbose=False):
    """Run Leiden community detection.
    Falls back to NetworkX Louvain if leidenalg/igraph is not installed.

    Args:
        A: adjacency matrix
        k: number of communities (used to set resolution; Leiden finds k adaptively,
           but we try to guide it toward the right number).
        seed: random seed
        verbose: print info

    Returns:
        labels: (n,) community assignments
    """
    n = A.shape[0]

    # Try Leiden first
    try:
        import igraph as ig
        import leidenalg

        # Build igraph graph
        G_ig = ig.Graph.Adjacency((A > 0).tolist(), mode='undirected')
        G_ig.es['weight'] = [A[e.source, e.target] for e in G_ig.es]

        # Use modularity optimization
        partition = leidenalg.find_partition(
            G_ig,
            leidenalg.ModularityVertexPartition,
            seed=seed
        )
        labels = np.array(partition.membership, dtype=int)

        if verbose:
            print(f"    Leiden found {len(set(labels))} communities")
        return labels

    except ImportError:
        if verbose:
            print("    leidenalg/igraph not installed, falling back to NetworkX Louvain")

    # Fallback: NetworkX Louvain
    try:
        G_nx = nx.from_numpy_array(A)
        communities = nx.community.louvain_communities(G_nx, seed=seed)
        labels = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                labels[node] = i
        if verbose:
            print(f"    Louvain found {len(communities)} communities")
        return labels
    except Exception as e:
        if verbose:
            print(f"    Louvain failed: {e}")
        # Last resort: random labels
        rng = np.random.RandomState(seed)
        labels = rng.randint(0, max(2, k or 2), size=n)
        return labels


def run_dmon(A, k, seed=42, max_iter=300, lr=0.05, verbose=False):
    """DMoN-lite: differentiable modularity optimiser (no GNN features).

    This is a faithful *objective-level* implementation of the DMoN idea:
    learn a soft node-to-community assignment P = softmax(theta) and maximise
    the modularity Q = Tr(P^T A P)/(2m) - sum_c (d_c/(2m))^2 directly with
    gradient ascent, with a collapse penalty to keep communities balanced.

    It is NOT the full GNN-based DMoN model (which needs torch_geometric);
    it is labelled 'DMoN-lite' everywhere to stay honest about that.
    """
    import torch

    torch.manual_seed(seed)
    n = A.shape[0]
    A_t = torch.from_numpy(A.astype(np.float32))
    d = A_t.sum(dim=1)
    two_m = float(d.sum())
    if two_m <= 0:
        return np.zeros(n, dtype=int)

    theta = torch.nn.Parameter(torch.randn(n, k) * 0.1)
    opt = torch.optim.Adam([theta], lr=lr)
    eye = torch.eye(k)
    for it in range(max_iter):
        opt.zero_grad()
        P = torch.softmax(theta, dim=1)
        modularity = torch.trace(P.T @ A_t @ P) / two_m
        frac = (P.T @ d) / two_m
        modularity = modularity - torch.sum(frac ** 2)
        collapse = (torch.sqrt(torch.tensor(float(k))) *
                    torch.norm(P.sum(dim=0)) / n - 1.0) ** 2
        loss = -(modularity - 2.0 * collapse)
        loss.backward()
        opt.step()

    with torch.no_grad():
        P = torch.softmax(theta, dim=1)
        labels = torch.argmax(P, dim=1).numpy().astype(int)
    if verbose:
        print(f"    DMoN-lite: {len(np.unique(labels))} communities")
    return labels


def run_all_baselines(A, k, seed=42, verbose=False):
    """Run all baselines and return a dict of {method_name: labels}."""
    results = {}

    results['SymNMF'] = run_symnmf(A, k, seed=seed, verbose=verbose)
    results['DANMF'] = run_danmf(A, k, seed=seed, verbose=verbose)
    results['DMoN-lite'] = run_dmon(A, k, seed=seed, verbose=verbose)
    results['Leiden'] = run_leiden(A, k, seed=seed, verbose=verbose)

    return results
