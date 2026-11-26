# Final project note — SG-DMNMF

This document supersedes the CP1 checkpoint PDF (`Team8_ArpitSahu_06October2026.pdf`),
which described the *planned* method. It records what was actually implemented,
what was fixed, and what the generated results show.

## 1. Implemented method

- Triangle motif `B = A ∘ (A@A)`, rescaled to `max(A)`; mixed matrix
  `M(α) = (1−α)A + αB`.
- Deep symmetric NMF `M ≈ Ψ W Ψᵀ`, `Ψ = H₁H₂…Hₚ`, non-negativity via softplus,
  Laplacian smoothness `λ·tr(ΨᵀLΨ)` and diagonal-dominance penalty `μ·‖W∘(1−I)‖²`.
- **Selection rule (contribution):** for each `α ∈ {0, .1, .3, .5, .7, .9}`,
  fit two 80%-edge-subsampled graphs, label by `argmax(Ψ)`, score by AMI
  between partitions; choose the most stable α (ties → modularity), refit on
  the full graph. Ground-truth labels are never used by the selection rule.
- Comparators: SymNMF, DANMF, DMoN-lite, Leiden, fixed α, no motif, and a
  label-using oracle (upper bound only).

## 2. Defects found and repaired

| Defect | Consequence | Fix |
|---|---|---|
| Unconstrained full `W` and badly scaled factor init (Ψ≈31, loss≈4×10⁹) | Both Ψ columns collapsed → random labels; Karate ACC 0.588 | Near-identity `W`, scale-aware init, NMF/spectral embedding warm start; Karate ACC 0.971 |
| `len(set(labels))` in runner | Crash whenever a dataset is unlabeled (e.g. real Dolphin) | `determine_k()` with modularity estimate |
| Multiplicative update overflow in SymNMF | Loss exploded at final iteration | Clipped factors |
| DMoN baseline missing | Report promised 4 baselines, code had 3 | `DMoN-lite` added (objective-level, no GNN) |
| Duplicate E4 configuration | Misleading ablation | Removed; `Depth=2` is the full model |
| Live-download-only datasets | Empty cache → no run possible anywhere offline | Offline-first loaders with clearly named synthetic stand-ins |
| Greek α in console output | UnicodeEncodeError on Windows cp1252 | ASCII method labels + UTF-8 stdout |
| No UI / README / tests | Not demonstrable | Flask dashboard, README, teacher guide, 23 unittest cases |

## 3. Results from the committed quick run (`--quick`, offline, 2 seeds)

| Dataset | Stability-selected α | Selected NMI | Oracle α / NMI | Best baseline |
|---|---|---|---|---|
| Karate | 0.7 | 0.837 | 0.0 / 0.837 | SymNMF 0.837 |
| Dolphin (real, unlabeled) | 0.5 | N/A (no labels) | skipped by design | Leiden Q=0.517 |
| Quick synthetic (offline) | 0.3 | 0.788 | 0.5 / 0.797 | SymNMF 0.763 |
| LFR-mu0.3 (real LFR) | 0.1 | 0.690 | 0.0 / 0.690 | SymNMF 0.712 |

Reading of the evidence:

- On the real LFR benchmark, stability selects α=0.1 and the label-using oracle
  selects α=0.0; both give NMI 0.690 — strong support for RQ1/RQ2. SymNMF is
  slightly higher (0.712), an honest negative point about the deep model.
- On the quick synthetic graph, stability peaks at 0.3 while the oracle is 0.5;
  the selected α is within 0.01 NMI of the oracle — good but not identical.
- The real Dolphin GML has no authentic community attribute, so it stays
  **unlabeled**: supervised metrics are N/A and modularity is used instead.
  No labels were invented. This also exercises the unsupervised branch of the
  runner, which previously crashed here.
- E4 on Dolphin shows modularity changes across configurations, so each model
  component has a measurable effect.

Full tables and figures are in `results/` and `figures/`; the dashboard
presents them with explanation.

## 4. Deviations from the CP1 report (state these openly)

1. The code uses the **raw mixed matrix**, not the degree-normalised
   `Â, M_norm` of the planned Algorithm 1 — symmetric NMF reconstructs the
   unnormalised matrix more meaningfully.
2. Regularisation weights are fixed (`λ=μ=1`) rather than tuned; Wilcoxon
   significance tests are future work; results are mean ± std over seeds.
3. DMoN is the simplified `DMoN-lite` (no GNN encoder). Cora/Citeseer require
   `torch-geometric`; otherwise clearly marked synthetic stand-ins are used.
4. The model is the symmetric "DMRA-lite", not the paper's asymmetric model.
   The novelty lies in the label-free selection rule, which is independent of
   that choice.

## 5. Reproduction

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/run_all.py --quick
python run_dashboard.py
```
