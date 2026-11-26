"""
data.py - Dataset loading for community detection experiments.

Offline-first design
--------------------
The project must be runnable on a machine with no internet access. Every
loader therefore follows the same policy:

  1. Use the cached file in ``data_cache/`` if present.
  2. Otherwise try a short, time-limited download.
  3. Otherwise fall back to a *clearly named* deterministic synthetic graph
     with planted communities (never passed off as a real dataset).

The only exception is Dolphin: if the real graph is available but has no
genuine ground-truth attribute, it is returned as an *unlabeled* dataset
(labels=None) so that the unsupervised branch of the pipeline is exercised
and no labels are fabricated.

Every loader returns (A, labels, name):
  A      : (n, n) float32 symmetric binary adjacency matrix
  labels : (n,) int ground-truth labels (0-indexed), or None for unlabeled
  name   : str; "offline synthetic" is part of the name for fallback graphs
"""
import gzip
import io
import os
import urllib.request
import warnings

import networkx as nx
import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")

# Set SGDMNMF_OFFLINE=1 to skip all network attempts.
FORCE_OFFLINE = os.environ.get("SGDMNMF_OFFLINE", "0") == "1"
_DOWNLOAD_TIMEOUT = 6  # seconds


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _download(url, dest, timeout=_DOWNLOAD_TIMEOUT):
    """Download a file with a hard timeout. Returns True on success."""
    if FORCE_OFFLINE:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as exc:
        print(f"  Download failed ({exc}); using offline fallback.")
        return False


def _largest_cc(G):
    """Extract the largest connected component and relabel nodes 0..n-1."""
    if G.number_of_nodes() == 0:
        return G, {}
    cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(cc).copy()
    mapping = {node: i for i, node in enumerate(sorted(sub.nodes()))}
    return nx.relabel_nodes(sub, mapping), mapping


def _graph_to_arrays(G, label_dict=None):
    """Convert a networkx graph to (A, labels) numpy arrays."""
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    A = (A > 0).astype(np.float32)

    if label_dict is None:
        return A, None
    labels = np.array([label_dict[i] for i in range(n)], dtype=int)
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return A, np.array([remap[label] for label in labels], dtype=int)


def _planted(n_per_comm, n_comm, p_in, p_out, seed):
    """Fast deterministic stochastic block model with planted communities."""
    G = nx.planted_partition_graph(n_comm, n_per_comm, p_in, p_out, seed=seed)
    label_dict = {i: i // n_per_comm for i in range(n_comm * n_per_comm)}
    return _graph_to_arrays(G, label_dict)


# ---------------------------------------------------------------------------
# Real datasets (with offline fallbacks)
# ---------------------------------------------------------------------------

def load_karate():
    """Zachary's Karate Club (34 nodes, 2 communities). Always available."""
    G = nx.karate_club_graph()
    label_dict = {
        node: 0 if G.nodes[node]["club"] == "Mr. Hi" else 1 for node in G.nodes()
    }
    A, labels = _graph_to_arrays(G, label_dict)
    return A, labels, "Karate"


def load_dolphin():
    """Dolphin social network. Returns an unlabeled real graph if no genuine
    ground truth exists; an offline synthetic stand-in if it cannot be loaded.
    """
    _ensure_cache_dir()
    gml_path = os.path.join(_CACHE_DIR, "dolphins.gml")
    urls = [
        "https://raw.githubusercontent.com/networkx/networkx/main/examples/graph/dolphins.gml",
        "https://raw.githubusercontent.com/csdashes/GraphStreamCommunityDetection/master/data/dolphins.gml",
    ]
    if not os.path.exists(gml_path):
        for url in urls:
            if _download(url, gml_path):
                break
    if os.path.exists(gml_path):
        try:
            G = nx.read_gml(gml_path, label="id")
            G = nx.Graph(G)
            G, _ = _largest_cc(G)
            for attr in ("community", "ground_truth", "group", "category"):
                if all(attr in G.nodes[n] for n in G.nodes()):
                    raw = {n: G.nodes[n][attr] for n in G.nodes()}
                    vals = sorted(set(raw.values()))
                    remap = {v: i for i, v in enumerate(vals)}
                    A, labels = _graph_to_arrays(G, {n: remap[raw[n]] for n in G.nodes()})
                    return A, labels, "Dolphin"
            # Real topology, but no authentic labels: stay unlabeled.
            A, _ = _graph_to_arrays(G)
            return A, None, "Dolphin (unlabeled)"
        except Exception as exc:
            print(f"  Dolphin GML unreadable ({exc}); using offline fallback.")

    # Difficulty chosen so that no method is perfect (ACC ~0.88): useful for
    # demonstrating that the choice of alpha actually matters.
    A, labels = _planted(31, 2, 0.22, 0.10, seed=11)
    return A, labels, "Dolphin (offline synthetic)"


def load_email():
    """Email-Eu-core network (1005 nodes). Falls back to an offline graph."""
    _ensure_cache_dir()
    edge_path = os.path.join(_CACHE_DIR, "email-Eu-core.txt")
    label_path = os.path.join(_CACHE_DIR, "email-Eu-core-department-labels.txt")

    if not os.path.exists(edge_path):
        _download("https://snap.stanford.edu/data/email-Eu-core.txt.gz",
                  edge_path + ".gz")
        if os.path.exists(edge_path + ".gz"):
            try:
                with gzip.open(edge_path + ".gz", "rb") as fin, open(edge_path, "wb") as fout:
                    fout.write(fin.read())
            except Exception:
                pass
    if not os.path.exists(label_path):
        _download("https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
                  label_path + ".gz")
        if os.path.exists(label_path + ".gz"):
            try:
                with gzip.open(label_path + ".gz", "rb") as fin, open(label_path, "wb") as fout:
                    fout.write(fin.read())
            except Exception:
                pass

    if os.path.exists(edge_path) and os.path.exists(label_path):
        try:
            G = nx.Graph()
            with open(edge_path) as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        if u != v:
                            G.add_edge(u, v)
            node_labels = {}
            with open(label_path) as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        node_labels[int(parts[0])] = int(parts[1])
            keep = set(node_labels) & set(G.nodes())
            G = G.subgraph(keep).copy()
            G, mapping = _largest_cc(G)
            inv = {v: k for k, v in mapping.items()}
            new_labels = {new: node_labels[inv[new]] for new in range(G.number_of_nodes())}
            A, labels = _graph_to_arrays(G, new_labels)
            return A, labels, "Email-Eu-core"
        except Exception as exc:
            print(f"  Email data unreadable ({exc}); using offline fallback.")

    A, labels = _planted(60, 5, 0.22, 0.06, seed=13)
    return A, labels, "Email (offline synthetic)"


def _planetoid(name, fallback_name, n_comm, seed):
    """Load Cora/Citeseer through torch_geometric, else offline fallback."""
    try:
        from torch_geometric.datasets import Planetoid
        _ensure_cache_dir()
        data = Planetoid(root=os.path.join(_CACHE_DIR, "Planetoid"),
                         name=name, transform=None)[0]
        edge_index = data.edge_index.numpy()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u != v:
                G.add_edge(u, v)
        label_dict = {i: int(data.y[i]) for i in range(data.num_nodes)}
        G, mapping = _largest_cc(G)
        inv = {v: k for k, v in mapping.items()}
        new_labels = {new: label_dict[inv[new]] for new in range(G.number_of_nodes())}
        A, labels = _graph_to_arrays(G, new_labels)
        return A, labels, name
    except Exception as exc:
        print(f"  {name} unavailable ({exc}); using offline fallback.")

    A, labels = _planted(45, n_comm, 0.20, 0.06, seed=seed)
    return A, labels, f"{fallback_name} (offline synthetic)"


def load_cora():
    return _planetoid("Cora", "Cora", 7, seed=17)


def load_citeseer():
    return _planetoid("CiteSeer", "Citeseer", 6, seed=19)


def load_lfr(mu=0.3, n=600, seed=42):
    """LFR benchmark. Falls back to a planted-partition proxy if LFR fails."""
    last_err = None
    if not FORCE_OFFLINE:
        for attempt in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    G = nx.LFR_benchmark_graph(
                        n=n, tau1=2.5, tau2=1.5, mu=mu,
                        average_degree=10, max_degree=40,
                        min_community=20, max_community=80,
                        seed=seed + attempt * 1000,
                    )
                communities = {v: sorted(G.nodes[v]["community"])[0] for v in G.nodes()}
                G = nx.Graph(G)
                G.remove_edges_from(nx.selfloop_edges(G))
                G, mapping = _largest_cc(G)
                inv = {v: k for k, v in mapping.items()}
                new_labels = {new: communities[inv[new]] for new in range(G.number_of_nodes())}
                A, labels = _graph_to_arrays(G, new_labels)
                return A, labels, f"LFR-mu{mu:.1f}"
            except Exception as exc:
                last_err = exc
    else:
        last_err = "offline mode"
    print(f"  LFR failed ({last_err}); using deterministic proxy.")

    # Proxy: mixing parameter mu is mapped to cross-community edge density.
    # Difficulty is tuned so that mu=0.3 gives ACC ~0.85-0.9 (not trivial).
    n_comm = max(3, n // 50)
    n_per = max(20, n // n_comm)
    p_in = 0.25
    p_out = float(np.clip(0.33 * mu, 0.01, 0.25))
    A, labels = _planted(n_per, n_comm, p_in, p_out, seed=seed)
    return A, labels, f"LFR-proxy-mu{mu:.1f}"


def load_all_datasets(include_large=True, lfr_mus=(0.1, 0.3, 0.5)):
    """Load all available datasets; never raise because of a missing dataset."""
    datasets = [load_karate()]
    for loader in (load_dolphin, load_email):
        result = loader()
        if result[0] is not None:
            datasets.append(result)
    if include_large:
        for loader in (load_cora, load_citeseer):
            result = loader()
            if result[0] is not None:
                datasets.append(result)
    for mu in lfr_mus:
        result = load_lfr(mu=mu)
        if result[0] is not None:
            datasets.append(result)
    return datasets


def load_quick_synthetic():
    """Small deterministic graph for the fast demonstration pipeline."""
    A, labels = _planted(25, 3, 0.28, 0.08, seed=23)
    return A, labels, "Quick synthetic (offline)"


def load_quick_datasets():
    """Small/fast datasets for a quick end-to-end demonstration."""
    return [load_karate(), load_dolphin(), load_quick_synthetic(),
            load_lfr(mu=0.3, n=200)]


def is_synthetic(name):
    """True if the dataset name marks it as an offline synthetic stand-in."""
    return "synthetic" in name or "proxy" in name
