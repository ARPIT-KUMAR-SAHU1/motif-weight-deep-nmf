# SG-DMNMF — Stability-Guided Deep Motif NMF for Community Detection

A complete, runnable minor-project implementation of **stability-guided selection of
the motif weight in deep NMF-based community detection**. The core question:

> The base paper (DMRA-NMF, Sohrabi et al., EAAI 2026) mixes edges with triangle
> motifs using a weight α that it tunes **using ground-truth labels**. How can we
> choose α on a network whose labels we do not have?

Our answer: choose α by **perturbation stability**. Remove 20% of edges twice,
cluster both perturbed graphs, and measure partition agreement with Adjusted
Mutual Information (AMI). The α with the highest stability wins. No labels are
read at any point.

---

## 1. Quick start

```bash
pip install -r requirements.txt

python -m unittest discover -s tests -v      # 23 sanity tests
python experiments/run_all.py --quick        # generates results/ and figures/
python run_dashboard.py                      # http://127.0.0.1:8000
```

One-command alternative on Windows: double-click **`start_project.bat`**.
Cross-platform: `python run_project.py all` (tests → quick run → dashboard).

For the full protocol (more seeds/iterations, all datasets):
`python experiments/run_all.py`

Set `SGDMNMF_OFFLINE=1` to force offline synthetic data on any machine.

---

## 2. Project structure

| Path | Purpose |
|---|---|
| `src/models.py` | `SymNMF`, `DMRALiteModel`, `DMRALite` — the community detection models |
| `src/motif.py` | Triangle motif `B = A ∘ (A@A)`, mixing, triangle counting |
| `src/alpha_select.py` | Stability-guided α selection (the contribution), oracle and fixed comparators |
| `src/data.py` | Offline-first dataset loaders (real when available, clearly-labelled synthetic otherwise) |
| `src/evaluate.py` | Hungarian ACC, NMI, ARI, AMI, modularity |
| `src/baselines.py` | SymNMF, DANMF, **DMoN-lite**, Leiden/Louvain |
| `experiments/run_all.py` | E1 clean benchmark, E2 α sensitivity, E3 robustness, E4 ablation |
| `tests/` | unittest suite: motif, models, metrics, data, stability |
| `ui/` | Flask dashboard (templates, CSS, JS) |
| `results/` | Generated CSVs + `summary.json` |
| `figures/` | Generated plots |
| `docs/TEACHER_GUIDE.md` | Explanation, demo script, and likely viva questions |
| `Team8_...pdf` | CP1 checkpoint report (the original design document) |

---

## 3. Method — explainable in two minutes

1. **Multi-order association.** From adjacency `A`, compute the triangle motif
   matrix `B[i,j] = number of triangles containing edge (i,j)` using
   `B = A ∘ (A@A)`. Rescale `B` so `max(B) = max(A)` and mix:
   `M(α) = (1−α)A + αB`.

2. **Deep factorisation.** Approximate `M ≈ Ψ W Ψᵀ`, where `Ψ = H₁H₂…Hₚ`
   (all factors non-negative via softplus). The loss is
   `‖M − ΨWΨᵀ‖² + λ·tr(ΨᵀLΨ) + μ·‖W ∘ (1−I)‖²`:
   reconstruction, graph-Laplacian smoothness, and a diagonal-dominance
   penalty on `W` that encourages disjoint communities. Labels are
   `argmax_k Ψ[i,k]`.

3. **Stability-guided α (our contribution).** For each candidate
   `α ∈ {0, .1, .3, .5, .7, .9}`, generate two independent 80%-edge subsamples,
   fit the model on each, and compute `AMI(C₁, C₂)`. Select
   `α* = argmax stability(α)`; ties are broken by full-graph modularity.
   Finally refit on the complete graph with α*.

4. **Comparators.** Fixed α=0.5, no motif α=0, and an *oracle* α chosen with
   labels (an upper bound, never used as a real method).

### Important implementation note
This is a **symmetric, triangle-only, Adam/softplus** simplification named
**DMRA-lite**, not an exact reproduction of the asymmetric paper. The
experiments are therefore evidence for the *selection rule*, not a claim of
matching the published numbers. The README and dashboard say this explicitly.

---

## 4. What was corrected from the initial codebase

| Issue found | Fix applied |
|---|---|
| Deep model collapsed to a rank‑1 factorisation (both Ψ columns identical): loss decreased but labels were random (Karate ACC 0.588 vs 0.971 for SymNMF). | `W` initialised near the identity; factors initialised from an NMF/spectral embedding so Ψ starts with community structure. Karate now reaches ACC 0.971, NMI 0.837. |
| Massive initial scale mismatch (Ψ ≈ 31, loss ≈ 4×10⁹) wasted all training iterations. | Scale-aware factor initialisation (`Ψ ≈ 0.5`). |
| `run_all.py` crashed on unlabeled datasets: `len(set(labels))` with `labels=None`. | Uses `determine_k()`, which derives k by modularity optimisation and never invents labels. |
| SymNMF multiplicative updates overflowed on the last iterations. | Clipped/normalised factors; no more NaN/Inf. |
| DMoN baseline listed in the report but missing in code. | Added `DMoN-lite` (differentiable modularity with collapse penalty), clearly marked as not the GNN version. |
| E4 contained two identical configurations. | Duplicate removed; `Depth=2` is the full-model reference. |
| Experiments required internet; empty cache meant nothing could run. | Offline-first loaders with deterministic planted-community fallbacks, clearly named `offline synthetic`. |
| No UI, no README, no tests beyond motif. | Flask dashboard, this README, `docs/TEACHER_GUIDE.md`, and a 20-test suite. |
| Report claims degree-normalised matrices and Wilcoxon tests that the code did not implement. | Documented honestly in `docs/TEACHER_GUIDE.md`; code uses raw mixed matrices (appropriate for symmetric NMF) and reports mean ± std. |

---

## 5. Experiments

- **E1 — Clean benchmark.** All methods × all datasets × seeds; ACC/NMI/ARI/AMI/modularity tables in `results/e1_clean_benchmark.csv`.
- **E2 — α sensitivity.** Metric-vs-α and stability-vs-α plots (`figures/e2_*.png`) plus `results/e2_alpha_sensitivity.csv`.
- **E3 — Robustness.** Edge deletion, edge addition, triangle-targeted deletion at 10–40% noise (`figures/e3_*.png`, `results/e3_robustness.csv`).
- **E4 — Ablation.** No motif, no Laplacian, no diagonal penalty, depth 1/2/3 (`results/e4_ablation.csv`).

`results/summary.json` records the timestamp, datasets, environment and method
description for the dashboard.

---

## 6. Data policy (important for honesty)

Real datasets are loaded from `data_cache/` when present, or downloaded with a
short timeout when possible. When offline, deterministic planted-partition
graphs are generated instead and their names contain **"offline synthetic"** or
**"LFR-proxy"**. They exist so the whole pipeline can be demonstrated offline;
they must not be used to claim state-of-the-art performance. Dolphin, when the
real graph is available but has no authentic community attribute, is returned
**unlabeled** (`labels=None`) and is handled by the unsupervised branch of the
code — no labels are fabricated.

---

## 7. Dashboard

`python run_dashboard.py` starts a local server at `http://127.0.0.1:8000` and
opens a browser. It shows:

- the method as four short steps and an algorithm box,
- dataset metadata with real/synthetic and labeled/unlabeled badges,
- E1 and E4 result tables with the best ACC row highlighted,
- a gallery of all generated figures and links to raw CSVs,
- a "how to present this" checklist.

It is read-only: it never re-runs experiments.

---

## 8. Limitations (state these openly)

- Symmetric factorisation and triangle motifs only — no asymmetric/directed
  graphs and no higher-order motifs beyond triangles.
- k is assumed known (from labels when available, else modularity estimate).
- DMoN-lite optimises the modularity objective without a GNN encoder.
- Offline synthetic stand-ins are demonstrations, not benchmarks.
- The CP1 PDF describes the *planned* method (degree normalisation, Wilcoxon
  tests); the implemented project uses raw mixed matrices and mean ± std.
  `docs/TEACHER_GUIDE.md` explains the discrepancy.

---

## 9. Troubleshooting

- **Port 8000 busy:** `PORT=8080 python run_dashboard.py`.
- **Slow full run:** use `--quick`; large LFR graphs are the expensive part.
- **Datasets skipped:** expected offline; check the dataset badges in the UI.
- **`torch_geometric` missing:** Cora/Citeseer use the offline fallback.
