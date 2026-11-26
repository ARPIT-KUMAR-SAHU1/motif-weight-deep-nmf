"""
run_all.py - Main experiment runner.

Experiments:
  E1: Clean benchmark (all methods on all datasets)
  E2: Alpha sensitivity (score vs alpha per dataset)
  E3: Robustness (edge noise: deletion, addition, triangle-targeted deletion)
  E4: Ablation (no motif, no Laplacian, no diagonal penalty, depth 1/2/3)

Usage:
  python experiments/run_all.py              # full run (a few minutes offline)
  python experiments/run_all.py --quick      # quick run (~1-2 min)
  python experiments/run_all.py --e1         # run only E1
  python experiments/run_all.py --e2         # run only E2
  python experiments/run_all.py --e3         # run only E3
  python experiments/run_all.py --e4         # run only E4
"""
import sys
import os
import json
import argparse
import warnings
import platform
from datetime import datetime
warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which cannot print some Unicode glyphs.
# Force UTF-8 with a safe fallback so a full run never crashes at a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from src.utils import set_seed, Timer, print_table
from src.data import (
    load_karate, load_dolphin, load_email, load_cora, load_citeseer,
    load_lfr, load_quick_synthetic, is_synthetic,
)
from src.motif import triangle_motif_matrix, count_triangles
from src.models import DMRALite
from src.baselines import run_symnmf, run_danmf, run_leiden, run_dmon
from src.alpha_select import (
    select_alpha_stability, select_alpha_oracle,
    CANDIDATE_ALPHAS, subsample_edges
)
from src.evaluate import evaluate_all

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_datasets(quick=False):
    """Load datasets based on quick/full mode."""
    datasets = []

    # Always: Karate (sanity only)
    datasets.append(load_karate())

    # Dolphin
    r = load_dolphin()
    if r[0] is not None:
        datasets.append(r)

    if quick:
        # Small deterministic graph so the whole pipeline runs in minutes.
        datasets.append(load_quick_synthetic())
    else:
        # Email
        r = load_email()
        if r[0] is not None:
            datasets.append(r)

        # Cora
        r = load_cora()
        if r[0] is not None:
            datasets.append(r)

        # Citeseer
        r = load_citeseer()
        if r[0] is not None:
            datasets.append(r)

    # LFR
    mus = [0.3] if quick else [0.1, 0.3, 0.5]
    for mu in mus:
        r = load_lfr(mu=mu, n=200 if quick else 400)
        if r[0] is not None:
            datasets.append(r)

    return datasets


def determine_k(A: np.ndarray, labels: np.ndarray = None, name: str = "") -> tuple:
    """Determine the community count k for a dataset.
    
    Returns:
        (k, is_labeled):
          k: int, number of communities to use in models
          is_labeled: bool, True if k came from genuine ground-truth labels,
                            False if k is an unsupervised topological estimate.
    
    Logic:
      1. If genuine labels are provided (labels is not None):
         k = len(np.unique(labels)) -> Ground truth k.
      2. If labels is None (unlabeled dataset):
         We do NOT invent ground truth. We determine k using an unsupervised
         modularity optimization pass (NetworkX Louvain / Leiden) on the graph topology.
         For Dolphin specifically, this finds its natural modularity-driven community
         count without human label invention.
    """
    if labels is not None:
        k = len(np.unique(labels))
        return k, True

    # Unsupervised estimate via modularity optimization
    from src.baselines import run_leiden
    est_labels = run_leiden(A, seed=42)
    k = max(2, len(np.unique(est_labels)))
    print(f"  [{name}] Unlabeled dataset (labels=None). Using unsupervised modularity estimate k={k}")
    return k, False


# ---------------------------------------------------------------------------
# E1: Clean Benchmark
# ---------------------------------------------------------------------------
def run_e1(datasets, n_seeds=5, quick=False, max_iter=200):
    """E1: Run all methods on all datasets, n_seeds repetitions."""
    if quick:
        n_seeds = 2
        max_iter = 100

    print("\n" + "="*70)
    print("  E1: CLEAN BENCHMARK")
    print("="*70)

    all_rows = []

    for A, labels, name in datasets:
        n = A.shape[0]
        k, is_labeled = determine_k(A, labels, name)
        n_tri = count_triangles(A)
        label_info = f"k={k} (ground truth)" if is_labeled else f"k={k} (unsupervised estimate, UNLABELED)"
        print(f"\n--- {name} (n={n}, {label_info}, triangles={n_tri}) ---")

        # Collect per-seed results for each method
        method_results = {}

        for seed in range(n_seeds):
            set_seed(seed)

            # 1. SymNMF
            pred = run_symnmf(A, k, seed=seed, max_iter=200)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault('SymNMF', []).append(m)

            # 2. DANMF (alpha=0, mu=0)
            pred = run_danmf(A, k, seed=seed, max_iter=max_iter)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault('DANMF', []).append(m)

            # 3. DMoN-lite (differentiable modularity)
            pred = run_dmon(A, k, seed=seed, max_iter=max_iter)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault('DMoN-lite', []).append(m)

            # 4. Leiden
            pred = run_leiden(A, k, seed=seed)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault('Leiden', []).append(m)

            # 4. DMRA-lite (fixed alpha=0.5)
            model = DMRALite(k=k, alpha=0.5, max_iter=max_iter, seed=seed)
            pred = model.fit(A)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault('DMRA-fixed-0.5', []).append(m)

            # 5. DMRA-lite (alpha=0, no motif)
            model = DMRALite(k=k, alpha=0.0, max_iter=max_iter, seed=seed)
            pred = model.fit(A)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault('DMRA-no-motif', []).append(m)

        # 6. Stability-selected alpha (run once, then evaluate with n_seeds)
        print(f"  Selecting alpha via stability...")
        stab_repeats = 3 if quick else 5
        stab_iter = 100 if quick else 200
        best_alpha, stab_scores = select_alpha_stability(
            A, k, n_repeats=stab_repeats, max_iter=stab_iter,
            base_seed=42, verbose=True
        )
        for seed in range(n_seeds):
            model = DMRALite(k=k, alpha=best_alpha, max_iter=max_iter, seed=seed)
            pred = model.fit(A)
            m = evaluate_all(labels, pred, A=A)
            method_results.setdefault(f'DMRA-stab(a={best_alpha:.1f})', []).append(m)

        # 7. Oracle alpha (pick best alpha using labels -- ONLY for labeled datasets)
        if is_labeled:
            print(f"  Finding oracle alpha...")
            oracle_alpha, oracle_scores = select_alpha_oracle(
                A, labels, k, max_iter=max_iter, seed=42, verbose=True
            )
            if oracle_alpha is not None:
                for seed in range(n_seeds):
                    model = DMRALite(k=k, alpha=oracle_alpha, max_iter=max_iter, seed=seed)
                    pred = model.fit(A)
                    m = evaluate_all(labels, pred, A=A)
                    method_results.setdefault(f'DMRA-oracle(a={oracle_alpha:.1f})', []).append(m)
        else:
            print(f"  Oracle alpha skipped: {name} is unlabeled (no ground-truth labels).")

        # Print summary table
        headers = ["Method", "ACC", "NMI", "ARI", "AMI", "Modularity"]
        rows = []
        for mname, mlist in method_results.items():
            mean_vals = {}
            std_vals = {}
            for metric in ['ACC', 'NMI', 'ARI', 'AMI', 'Modularity']:
                vals = [m[metric] for m in mlist]
                if np.all(np.isnan(vals)):
                    mean_vals[metric] = np.nan
                    std_vals[metric] = np.nan
                else:
                    mean_vals[metric] = np.nanmean(vals)
                    std_vals[metric] = np.nanstd(vals)
            row = [mname]
            for metric in ['ACC', 'NMI', 'ARI', 'AMI', 'Modularity']:
                if np.isnan(mean_vals[metric]):
                    row.append("N/A")
                else:
                    row.append(f"{mean_vals[metric]:.3f}±{std_vals[metric]:.3f}")
            rows.append(row)

            # Save to all_rows for CSV
            all_rows.append({
                'Dataset': name, 'Method': mname, 'IsLabeled': is_labeled,
                'IsSynthetic': is_synthetic(name),
                'ACC_mean': mean_vals['ACC'], 'ACC_std': std_vals['ACC'],
                'NMI_mean': mean_vals['NMI'], 'NMI_std': std_vals['NMI'],
                'ARI_mean': mean_vals['ARI'], 'ARI_std': std_vals['ARI'],
                'AMI_mean': mean_vals['AMI'], 'AMI_std': std_vals['AMI'],
                'Modularity_mean': mean_vals['Modularity'], 'Modularity_std': std_vals['Modularity'],
            })

        title_suffix = "LABELED" if is_labeled else "UNLABELED"
        print_table(headers, rows, title=f"E1 Results: {name} ({title_suffix})")

    # Save to CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "e1_clean_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved E1 results to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# E2: Alpha Sensitivity
# ---------------------------------------------------------------------------
def run_e2(datasets, max_iter=200, quick=False):
    """E2: Plot metric vs alpha for each dataset."""
    if quick:
        max_iter = 100

    print("\n" + "="*70)
    print("  E2: ALPHA SENSITIVITY")
    print("="*70)

    all_rows = []

    for A, labels, name in datasets:
        if 'Karate' in name:
            continue  # skip Karate (too easy, all alphas give same result)

        n = A.shape[0]
        k, is_labeled = determine_k(A, labels, name)
        print(f"\n--- {name} (n={n}, k={k}, is_labeled={is_labeled}) ---")

        alphas = CANDIDATE_ALPHAS
        metrics_per_alpha = {a: [] for a in alphas}

        n_seeds = 2 if quick else 3
        for seed in range(n_seeds):
            for alpha in alphas:
                model = DMRALite(k=k, alpha=alpha, max_iter=max_iter, seed=seed)
                pred = model.fit(A)
                m = evaluate_all(labels, pred, A=A)
                metrics_per_alpha[alpha].append(m)

        # Compute stability scores too
        stab_repeats = 2 if quick else 3
        stab_iter = 60 if quick else 120
        _, stab_scores = select_alpha_stability(
            A, k, n_repeats=stab_repeats, max_iter=stab_iter,
            base_seed=42, verbose=False
        )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: clustering metrics vs alpha (supervised if labeled, modularity if unlabeled)
        ax = axes[0]
        if is_labeled:
            for metric in ['ACC', 'NMI', 'ARI', 'AMI']:
                means = [np.mean([m[metric] for m in metrics_per_alpha[a]]) for a in alphas]
                stds = [np.std([m[metric] for m in metrics_per_alpha[a]]) for a in alphas]
                ax.errorbar(alphas, means, yerr=stds, marker='o', label=metric, capsize=3)
            ax.set_ylabel('Metric value')
            ax.set_title(f'{name}: Clustering metrics vs alpha')
        else:
            means = [np.mean([m['Modularity'] for m in metrics_per_alpha[a]]) for a in alphas]
            stds = [np.std([m['Modularity'] for m in metrics_per_alpha[a]]) for a in alphas]
            ax.errorbar(alphas, means, yerr=stds, marker='o', color='purple', label='Modularity (Q)', capsize=3)
            ax.set_ylabel('Modularity (Q)')
            ax.set_title(f'{name} (Unlabeled): Modularity vs alpha')

        ax.set_xlabel('Alpha (motif weight)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: stability score vs alpha
        ax = axes[1]
        stab_alphas = sorted(stab_scores.keys())
        stab_vals = [stab_scores[a] for a in stab_alphas]
        ax.plot(stab_alphas, stab_vals, 'rs-', linewidth=2, markersize=8, label='Stability (AMI)')
        ax.set_xlabel('Alpha (motif weight)')
        ax.set_ylabel('Stability score')
        ax.set_title(f'{name}: Partition stability vs alpha')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"e2_alpha_sensitivity_{name}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved figure to {fig_path}")

        # Save data
        for alpha in alphas:
            metrics_to_log = ['ACC', 'NMI', 'ARI', 'AMI', 'Modularity'] if is_labeled else ['Modularity']
            for metric in metrics_to_log:
                vals = [m[metric] for m in metrics_per_alpha[alpha]]
                all_rows.append({
                    'Dataset': name, 'Alpha': alpha, 'Metric': metric,
                    'IsLabeled': is_labeled, 'IsSynthetic': is_synthetic(name),
                    'Mean': np.nanmean(vals), 'Std': np.nanstd(vals),
                    'Stability': stab_scores.get(alpha, np.nan)
                })

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "e2_alpha_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved E2 results to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# E3: Robustness
# ---------------------------------------------------------------------------
def add_random_edges(A, fraction, rng):
    """Add random edges (fraction of existing edge count)."""
    n = A.shape[0]
    num_existing = int(A.sum()) // 2
    num_add = int(num_existing * fraction)
    A_noisy = A.copy()
    added = 0
    attempts = 0
    while added < num_add and attempts < num_add * 10:
        i, j = rng.randint(0, n, size=2)
        if i != j and A_noisy[i, j] == 0:
            A_noisy[i, j] = 1.0
            A_noisy[j, i] = 1.0
            added += 1
        attempts += 1
    return A_noisy


def remove_random_edges(A, fraction, rng):
    """Remove random edges (fraction of existing edges)."""
    rows, cols = np.where(np.triu(A > 0, k=1))
    num_edges = len(rows)
    num_remove = int(num_edges * fraction)
    remove_idx = rng.choice(num_edges, size=min(num_remove, num_edges), replace=False)
    A_noisy = A.copy()
    for idx in remove_idx:
        i, j = rows[idx], cols[idx]
        A_noisy[i, j] = 0
        A_noisy[j, i] = 0
    return A_noisy


def remove_triangle_edges(A, fraction):
    """Remove edges that participate in the most triangles first."""
    B = A * (A @ A)  # motif matrix (unnormalized)
    np.fill_diagonal(B, 0)

    rows, cols = np.where(np.triu(A > 0, k=1))
    num_edges = len(rows)
    num_remove = int(num_edges * fraction)

    # Score each edge by triangle participation
    scores = np.array([B[rows[i], cols[i]] for i in range(num_edges)])
    # Remove highest-score edges first
    remove_idx = np.argsort(-scores)[:num_remove]

    A_noisy = A.copy()
    for idx in remove_idx:
        i, j = rows[idx], cols[idx]
        A_noisy[i, j] = 0
        A_noisy[j, i] = 0
    return A_noisy


def run_e3(datasets, max_iter=200, quick=False):
    """E3: Robustness under edge noise."""
    if quick:
        max_iter = 100

    print("\n" + "="*70)
    print("  E3: ROBUSTNESS UNDER EDGE NOISE")
    print("="*70)

    noise_levels = [0.1, 0.2, 0.3, 0.4]
    n_seeds = 2 if quick else 3
    all_rows = []

    for A, labels, name in datasets:
        if 'Karate' in name:
            continue
        n = A.shape[0]
        k, is_labeled = determine_k(A, labels, name)
        eval_metric = 'NMI' if is_labeled else 'Modularity'
        print(f"\n--- {name} (n={n}, k={k}, is_labeled={is_labeled}, robust_metric={eval_metric}) ---")

        # First get the stability-selected alpha on the CLEAN graph
        stab_repeats = 2 if quick else 3
        stab_iter = 60 if quick else 120
        stab_alpha, _ = select_alpha_stability(
            A, k, n_repeats=stab_repeats, max_iter=stab_iter,
            base_seed=42, verbose=False
        )
        print(f"  Stability alpha on clean graph: {stab_alpha:.1f}")

        alpha_configs = {
            'alpha=0 (no motif)': 0.0,
            'alpha=0.5 (fixed)': 0.5,
            f'alpha={stab_alpha:.1f} (stability)': stab_alpha,
        }

        # For each noise type
        for noise_type in ['edge_deletion', 'edge_addition', 'triangle_deletion']:
            print(f"  Noise type: {noise_type}")
            fig, ax = plt.subplots(figsize=(8, 5))

            for config_name, alpha in alpha_configs.items():
                metric_means = []
                metric_stds = []
                for noise_frac in noise_levels:
                    metric_vals = []
                    for seed in range(n_seeds):
                        rng = np.random.RandomState(seed)
                        if noise_type == 'edge_deletion':
                            A_noisy = remove_random_edges(A, noise_frac, rng)
                        elif noise_type == 'edge_addition':
                            A_noisy = add_random_edges(A, noise_frac, rng)
                        else:  # triangle_deletion
                            A_noisy = remove_triangle_edges(A, noise_frac)

                        model = DMRALite(k=k, alpha=alpha, max_iter=max_iter, seed=seed)
                        pred = model.fit(A_noisy)
                        m = evaluate_all(labels, pred, A=A_noisy)
                        metric_vals.append(m[eval_metric])

                        all_rows.append({
                            'Dataset': name, 'NoiseType': noise_type,
                            'NoiseFrac': noise_frac, 'Alpha': alpha,
                            'Config': config_name, 'Seed': seed,
                            'IsLabeled': is_labeled, 'IsSynthetic': is_synthetic(name),
                            'EvaluatedMetric': eval_metric,
                            'ACC': m['ACC'], 'NMI': m['NMI'],
                            'ARI': m['ARI'], 'AMI': m['AMI'],
                            'Modularity': m['Modularity'],
                        })

                    metric_means.append(np.nanmean(metric_vals))
                    metric_stds.append(np.nanstd(metric_vals))

                ax.errorbar(
                    [int(f*100) for f in noise_levels],
                    metric_means, yerr=metric_stds,
                    marker='o', label=config_name, capsize=3
                )

            ax.set_xlabel('Noise level (%)')
            ax.set_ylabel(eval_metric)
            ax.set_title(f'{name}: Robustness under {noise_type} ({eval_metric})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig_path = os.path.join(FIGURES_DIR, f"e3_robustness_{name}_{noise_type}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved {fig_path}")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "e3_robustness.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved E3 results to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# E4: Ablation
# ---------------------------------------------------------------------------
def run_e4(datasets, max_iter=200, quick=False):
    """E4: Ablation study -- effect of each component."""
    if quick:
        max_iter = 100

    print("\n" + "="*70)
    print("  E4: ABLATION STUDY")
    print("="*70)

    n_seeds = 2 if quick else 3
    all_rows = []

    ablation_configs = {
        # 'Depth=2' below is the full model; do not add a duplicate row.
        'No motif (a=0)':     dict(alpha=0.0, lam=1.0, mu=1.0, depth=2),
        'No Laplacian':       dict(alpha=0.5, lam=0.0, mu=1.0, depth=2),
        'No diagonal pen.':   dict(alpha=0.5, lam=1.0, mu=0.0, depth=2),
        'Depth=1':            dict(alpha=0.5, lam=1.0, mu=1.0, depth=1),
        'Depth=2':            dict(alpha=0.5, lam=1.0, mu=1.0, depth=2),
        'Depth=3':            dict(alpha=0.5, lam=1.0, mu=1.0, depth=3),
    }

    for A, labels, name in datasets:
        if 'Karate' in name:
            continue
        n = A.shape[0]
        k, is_labeled = determine_k(A, labels, name)
        print(f"\n--- {name} (n={n}, k={k}, is_labeled={is_labeled}) ---")

        headers = ["Config", "ACC", "NMI", "ARI", "AMI", "Modularity"]
        table_rows = []

        for config_name, params in ablation_configs.items():
            metrics_list = []
            for seed in range(n_seeds):
                model = DMRALite(
                    k=k,
                    alpha=params['alpha'],
                    lam=params['lam'],
                    mu=params['mu'],
                    depth=params['depth'],
                    max_iter=max_iter,
                    seed=seed
                )
                pred = model.fit(A)
                m = evaluate_all(labels, pred, A=A)
                metrics_list.append(m)

                all_rows.append({
                    'Dataset': name, 'Config': config_name, 'Seed': seed,
                    'IsLabeled': is_labeled, 'IsSynthetic': is_synthetic(name),
                    **m, **params
                })

            mean_vals = {}
            std_vals = {}
            for metric in ['ACC', 'NMI', 'ARI', 'AMI', 'Modularity']:
                vals = [m[metric] for m in metrics_list]
                if np.all(np.isnan(vals)):
                    mean_vals[metric] = np.nan
                    std_vals[metric] = np.nan
                else:
                    mean_vals[metric] = np.nanmean(vals)
                    std_vals[metric] = np.nanstd(vals)

            row = [config_name]
            for metric in ['ACC', 'NMI', 'ARI', 'AMI', 'Modularity']:
                if np.isnan(mean_vals[metric]):
                    row.append("N/A")
                else:
                    row.append(f"{mean_vals[metric]:.3f}±{std_vals[metric]:.3f}")
            table_rows.append(row)

        print_table(headers, table_rows, title=f"E4 Ablation: {name}")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "e4_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved E4 results to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run SG-DMNMF experiments")
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer seeds, smaller datasets, ~5 min')
    parser.add_argument('--e1', action='store_true', help='Run only E1')
    parser.add_argument('--e2', action='store_true', help='Run only E2')
    parser.add_argument('--e3', action='store_true', help='Run only E3')
    parser.add_argument('--e4', action='store_true', help='Run only E4')
    args = parser.parse_args()

    ensure_dirs()

    run_specific = args.e1 or args.e2 or args.e3 or args.e4
    run_all = not run_specific

    max_iter = 100 if args.quick else 200

    print("Loading datasets...")
    with Timer("Dataset loading"):
        datasets = load_datasets(quick=args.quick)

    # FIX: Dolphin may legitimately have labels=None. Use the same determine_k
    # helper the experiments use instead of len(set(labels)), which crashed.
    print(f"\nLoaded {len(datasets)} datasets:")
    dataset_meta = []
    for A, labels, name in datasets:
        n = A.shape[0]
        k, is_labeled = determine_k(A, labels, name)
        n_edges = int(A.sum()) // 2
        n_tri = count_triangles(A)
        synthetic = is_synthetic(name)
        print(f"  {name}: n={n}, edges={n_edges}, k={k}, triangles={n_tri}, "
              f"{'labeled' if is_labeled else 'UNLABELED'}, "
              f"{'synthetic' if synthetic else 'real'}")
        dataset_meta.append({
            'name': name, 'n': n, 'edges': n_edges, 'k': k,
            'labeled': bool(is_labeled), 'synthetic': bool(synthetic),
            'triangles': n_tri,
        })

    ran = []
    if run_all or args.e1:
        ran.append('E1')
        with Timer("E1 Clean Benchmark"):
            run_e1(datasets, quick=args.quick, max_iter=max_iter)

    if run_all or args.e2:
        ran.append('E2')
        with Timer("E2 Alpha Sensitivity"):
            run_e2(datasets, quick=args.quick, max_iter=max_iter)

    if run_all or args.e3:
        ran.append('E3')
        with Timer("E3 Robustness"):
            run_e3(datasets, quick=args.quick, max_iter=max_iter)

    if run_all or args.e4:
        ran.append('E4')
        with Timer("E4 Ablation"):
            run_e4(datasets, quick=args.quick, max_iter=max_iter)

    # Machine-readable summary consumed by the web dashboard.
    import torch
    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'mode': 'quick' if args.quick else 'full',
        'experiments': ran,
        'datasets': dataset_meta,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': torch.__version__,
        },
        'method': {
            'title': 'Stability-Guided Deep Motif NMF (SG-DMNMF)',
            'idea': ('Choose the motif weight alpha by how stable the partition '
                     'is under 80% edge subsampling (AMI between two perturbed '
                     'runs), without using ground-truth labels.'),
            'candidates': CANDIDATE_ALPHAS,
            'caveat': ('Implementation is a symmetric, triangle-only, Adam-trained '
                       'simplification (DMRA-lite); offline synthetic datasets are '
                       'clearly marked and are for demonstration, not publication.'),
        },
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"  Results saved in: {RESULTS_DIR}")
    print(f"  Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
