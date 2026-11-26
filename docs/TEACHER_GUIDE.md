# Teacher guide — explaining and defending the project

## 1. One-paragraph pitch

> "The base paper DMRA-NMF combines ordinary edges with triangle motifs using a
> weight α, but it chooses α manually with the help of ground-truth labels. That
> is impossible on a real, unlabelled network. My project, SG-DMNMF, selects α
> from the *stability* of the clustering under random edge removal: I subsample
> the graph twice, fit the same deep NMF model on both copies, and measure the
> agreement of the two partitions using Adjusted Mutual Information. The α with
> the highest agreement is chosen without ever looking at labels. I then compare
> it against fixed α, no-motif, an oracle, and standard baselines, and test
> robustness to noise and an ablation of each model component."

## 2. Three-minute demo script

1. **Show `quick_test.py` output first.** Karate club: SymNMF and the corrected
   DMRA-lite both get ACC ≈ 0.97, NMI ≈ 0.84. Say: *"This is my regression test —
   before the fix, my deep model reconstructed the graph but returned random
   labels (ACC 0.59)."*
2. **Open the dashboard** (`python run_dashboard.py`). Walk top-to-bottom:
   problem → gap → idea → algorithm → experiments → results.
3. **E2 figure:** point to where stability peaks and where accuracy peaks. If
   they coincide, that is the main evidence for RQ1/RQ2.
4. **E3 figures:** compare `alpha=0` with the stability-selected α under noise.
5. **E4 table:** show that removing the motif or the diagonal penalty changes
   performance, so each component earns its place.
6. **End with the limitations card** — being explicit about simplifications is
   a strength, not a weakness.

## 3. Likely questions and good answers

**Q: Why AMI and not just accuracy?**
A: We cannot use accuracy — that needs labels. AMI compares two *predicted*
partitions, so it is fully unsupervised. We also report AMI instead of NMI for
evaluation because NMI is biased toward finer partitions when cluster sizes
are unequal (Jerdee et al., Nat Commun 2025).

**Q: How do you know k (number of communities)?**
A: When genuine labels exist, k is their count. For unlabeled graphs we estimate
k with modularity optimisation (Leiden/Louvain). We never invent labels.

**Q: Why deep NMF instead of one layer?**
A: A product of non-negative layers can represent complicated, hierarchical
community structure with fewer parameters, and prior work (DANMF, DMRA-NMF)
shows improved clustering.

**Q: Is this exactly the DMRA-NMF paper?**
A: No. Ours is a *symmetric* factorisation using triangle motifs only, trained
with Adam and softplus rather than the paper's asymmetric multiplicative rules.
The code is named DMRA-lite for that reason. The *selection rule* is the novel
part, and it is model-agnostic.

**Q: What if the graph has no triangles?**
A: Then B is zero, α has no effect, and stability is flat across candidates;
the tie-break uses modularity. This is handled without special-casing.

**Q: Are the offline synthetic datasets real results?**
A: No. They are clearly marked "offline synthetic" / "LFR-proxy" and exist so
the pipeline runs without internet. Real datasets replace them automatically
when cached or downloadable. We do not make performance claims from synthetic
stand-ins.

**Q: The CP1 report mentions degree normalisation and Wilcoxon tests. Where are they?**
A: Those were the *plan*. The implementation uses the raw mixed matrix because
a symmetric NMF reconstruction of a degree-normalised matrix is harder to
interpret and performed worse in early tests. Statistical testing is left as
future work; results are reported as mean ± standard deviation over seeds.

## 4. Mapping to the research questions

| RQ | Evidence in the project |
|---|---|
| RQ1: can stability select α without labels? | E2 stability curves and `select_alpha_stability()`; the selected α is compared with the oracle in E1. |
| RQ2: how close is it to the supervised oracle? | E1 rows `DMRA-stab(α=…)` vs `DMRA-oracle(α=…)`. |
| RQ3: is motif integration robust to edge noise? | E3 plots for edge deletion, edge addition and triangle-targeted deletion. |

## 5. Honest limitations to volunteer before being asked

- k is assumed known; estimating it automatically is future work.
- Undirected graphs and triangle motifs only.
- DMoN-lite lacks the GNN encoder of the original DMoN.
- Full-scale results on Cora/Citeseer require `torch-geometric` and are not
  available in an offline environment.
- Run time grows with graph size; `--quick` is used for the demo.
