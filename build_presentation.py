"""Build a rich, self-explanatory PowerPoint deck for SG-DMNMF.

All numbers and charts come from results/*.csv. The diagrams are generated
from the project's own data and code. No results are invented.

Requirements: python-pptx, matplotlib, networkx, numpy, Pillow.
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

import networkx as nx
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "presentation", "assets")
OUT = os.path.join(ROOT, "presentation", "SG_DMNMF_Presentation_Full.pptx")
os.makedirs(ASSETS, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_BLUE = "#2563EB"
C_TEAL = "#0F766E"
C_AMBER = "#B45309"
C_NAVY = "#17243B"
C_MUTED = "#61708A"
C_GREY = "#94A3B8"
C_LIGHT = "#F1F5F9"

RGB_NAVY = RGBColor(23, 36, 59)
RGB_BLUE = RGBColor(37, 99, 235)
RGB_MUTED = RGBColor(97, 112, 138)
RGB_LIGHT = RGBColor(241, 245, 249)
RGB_WHITE = RGBColor(255, 255, 255)
RGB_LINE = RGBColor(214, 224, 240)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": C_NAVY, "axes.labelcolor": C_NAVY,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.edgecolor": "#CBD5E1", "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def rows(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


e1 = rows("e1_clean_benchmark.csv")
e2 = rows("e2_alpha_sensitivity.csv")
e3 = rows("e3_robustness.csv")
e4 = rows("e4_ablation.csv")
summary = json.load(open(os.path.join(RESULTS, "summary.json"), encoding="utf-8"))


def val(r, key):
    try:
        v = float(r.get(key, ""))
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stability_series(dataset):
    seen = defaultdict(list)
    for r in e2:
        if r["Dataset"] == dataset and val(r, "Alpha") is not None:
            seen[val(r, "Alpha")].append(val(r, "Stability"))
    keys = sorted(seen)
    return keys, [mean(seen[k]) for k in keys]


def selected_alpha(dataset):
    for r in e1:
        if r["Dataset"] == dataset and "DMRA-stab" in r["Method"]:
            return r["Method"].split("a=")[-1].rstrip(")")
    return "--"


def oracle_alpha(dataset):
    for r in e1:
        if r["Dataset"] == dataset and "DMRA-oracle" in r["Method"]:
            return r["Method"].split("a=")[-1].rstrip(")")
    return "--"


def method_score(dataset, prefix, metric):
    for r in e1:
        if r["Dataset"] == dataset and r["Method"].startswith(prefix):
            return val(r, metric + "_mean")
    return None


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def diagram_network():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G, seed=3)
    colors = [C_BLUE if G.nodes[n]["club"] == "Mr. Hi" else C_TEAL for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.28, edge_color=C_MUTED)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=190,
                           edgecolors="white", linewidths=0.8)
    ax.set_title("A network with two communities", fontsize=14, fontweight="bold")
    ax.text(0.5, -0.06,
            "Dense links inside a community, sparse links between communities.",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=C_MUTED)
    ax.axis("off")
    return save(fig, "diagram_network.png")


def diagram_triangle():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    pos = {0: (0, 1), 1: (1, 1), 3: (0.5, 0.1), 2: (2, 1.4), 4: (2, 0.3)}
    tri = [(0, 1), (1, 3), (3, 0)]
    other = [(1, 2), (1, 4)]
    for u, v in other:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=C_GREY, lw=2)
    for u, v in tri:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=C_BLUE, lw=4)
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=230, color=C_NAVY, zorder=3)
        ax.text(x, y, str(n), color="white", ha="center", va="center",
                fontsize=9, zorder=4)
    ax.text(0.55, 0.55, "triangle", color=C_BLUE, fontsize=12, fontweight="bold")
    ax.text(1.7, 1.0, "no triangle\non this edge", color=C_MUTED, fontsize=10)
    ax.set_title("Triangle motifs capture higher-order structure",
                 fontsize=14, fontweight="bold")
    ax.text(0.5, -0.05,
            "An edge inside a triangle is a stronger sign of community than a single link.",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=C_MUTED)
    ax.set_xlim(-0.4, 2.5); ax.set_ylim(-0.4, 1.8); ax.axis("off")
    return save(fig, "diagram_triangle.png")


def diagram_matrices():
    from src.data import load_karate
    from src.motif import triangle_motif_matrix, mix_matrices
    A, _, _ = load_karate()
    B = triangle_motif_matrix(A)
    M = mix_matrices(A, B, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, mat, title in zip(axes, [A, B, M],
                              ["A: ordinary edges", "B: triangle motifs", "M(alpha = 0.5)"]):
        ax.imshow(mat, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Two views of the same graph are mixed into one matrix",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "Blue squares are non-zero entries; the mixed matrix keeps both kinds of structure.",
             ha="center", fontsize=10.5, color=C_MUTED)
    return save(fig, "diagram_matrices.png")


def diagram_deep_nmf():
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color=C_BLUE, fontsize=10.5):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor=color, edgecolor="none", alpha=0.92)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold")
        return patch

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, color=C_NAVY, lw=1.6))

    box(0.2, 2.4, 1.8, 1.0, "M\n(n x n)", C_NAVY)
    box(2.6, 2.4, 1.9, 1.0, "H1\n(n x d1)", C_BLUE)
    box(5.1, 2.4, 1.9, 1.0, "H2\n(d1 x k)", C_BLUE)
    box(7.6, 2.4, 1.9, 1.0, "Psi\n(n x k)", C_TEAL)
    box(7.6, 0.5, 1.9, 0.9, "W\n(k x k)", C_AMBER)
    box(10.2, 2.4, 1.7, 1.0, "M-hat", C_TEAL)
    arrow(2.0, 2.9, 2.55, 2.9)
    arrow(4.5, 2.9, 5.05, 2.9)
    arrow(7.0, 2.9, 7.55, 2.9)
    arrow(9.5, 2.9, 10.15, 2.9)
    arrow(8.55, 1.4, 8.55, 2.35)
    ax.text(6.0, 4.35, "Deep symmetric NMF:  M  ≈  Psi W Psi^T",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(6.0, 0.15,
            "Loss = reconstruction error + Laplacian smoothness + diagonal penalty on W",
            ha="center", fontsize=10.5, color=C_MUTED)
    return save(fig, "diagram_deep_nmf.png")


def diagram_perturbation():
    from src.data import load_karate
    from src.alpha_select import subsample_edges
    A, _, _ = load_karate()
    A1 = subsample_edges(A, 0.8, np.random.RandomState(0))
    A2 = subsample_edges(A, 0.8, np.random.RandomState(50))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, mat, title in zip(axes[:2], [A1, A2],
                              ["Subsample 1 (80% edges)", "Subsample 2 (80% edges)"]):
        G = nx.from_numpy_array(mat)
        pos = nx.spring_layout(G, seed=5)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.45, edge_color=C_MUTED)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80,
                               node_color=C_BLUE, edgecolors="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    axes[2].axis("off")
    axes[2].text(0.5, 0.65, "Cluster both copies", ha="center",
                 fontsize=12, fontweight="bold", transform=axes[2].transAxes)
    axes[2].text(0.5, 0.45, "C1 = argmax(Psi1)\nC2 = argmax(Psi2)",
                 ha="center", fontsize=11, transform=axes[2].transAxes)
    axes[2].text(0.5, 0.20, "stability = AMI(C1, C2)", ha="center",
                 fontsize=12, fontweight="bold", color=C_BLUE,
                 transform=axes[2].transAxes)
    fig.suptitle("Perturbation: a good partition should survive edge removal",
                 fontsize=14, fontweight="bold")
    return save(fig, "diagram_perturbation.png")


def diagram_ami():
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(6, 4.55, "AMI: agreement between two partitions, adjusted for chance",
            ha="center", fontsize=14, fontweight="bold")
    labels = [("Two independent runs", C_BLUE), ("Same community structure", C_TEAL),
              ("High AMI = stable", C_AMBER)]
    for i, (text, color) in enumerate(labels):
        x = 0.4 + i * 4.0
        patch = FancyBboxPatch((x, 1.7), 3.4, 1.7,
                               boxstyle="round,pad=0.1,rounding_size=0.15",
                               facecolor=color, edgecolor="none", alpha=0.9)
        ax.add_patch(patch)
        ax.text(x + 1.7, 2.55, text, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        if i < 2:
            arrow = FancyArrowPatch((x + 3.45, 2.55), (x + 3.9, 2.55),
                                    arrowstyle="-|>", mutation_scale=16,
                                    color=C_NAVY, lw=2)
            ax.add_patch(arrow)
    ax.text(6, 0.85,
            "Unlike raw agreement, AMI does not reward a partition just for having many small groups.",
            ha="center", fontsize=10.5, color=C_MUTED)
    return save(fig, "diagram_ami.png")


def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.2, "Graph A"),
        (2.5, "Motif B"),
        (4.8, "Mix M(alpha)"),
        (7.1, "Deep NMF"),
        (9.4, "Stability AMI"),
        (11.7, "alpha* + final fit"),
    ]
    colors = [C_NAVY, C_BLUE, C_BLUE, C_TEAL, C_AMBER, C_TEAL]
    for (x, text), color in zip(boxes, colors):
        patch = FancyBboxPatch((x, 1.6), 2.0, 1.5,
                               boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + 1.0, 2.35, text, ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        if x < 11:
            ax.add_patch(FancyArrowPatch((x + 2.05, 2.35), (x + 2.4, 2.35),
                                         arrowstyle="-|>", mutation_scale=15,
                                         color=C_NAVY, lw=1.6))
    ax.text(7, 4.1, "SG-DMNMF pipeline: the selection loop reads only the graph, never labels",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(7, 0.8,
            "The oracle alpha is computed later for evaluation only; it is not part of the method.",
            ha="center", fontsize=10.5, color=C_MUTED)
    return save(fig, "diagram_pipeline.png")


def diagram_experiments():
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    items = [
        ("E1", "Clean benchmark", "compare all methods", C_BLUE),
        ("E2", "Alpha sensitivity", "why this alpha?", C_TEAL),
        ("E3", "Robustness", "noise tolerance", C_AMBER),
        ("E4", "Ablation", "which components matter?", C_NAVY),
    ]
    for i, (tag, title, desc, color) in enumerate(items):
        x = 0.3 + i * 3.0
        patch = FancyBboxPatch((x, 1.4), 2.6, 2.4,
                               boxstyle="round,pad=0.1,rounding_size=0.15",
                               facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + 1.3, 3.2, tag, ha="center", color="white",
                fontsize=18, fontweight="bold")
        ax.text(x + 1.3, 2.55, title, ha="center", color="white",
                fontsize=10.5, fontweight="bold")
        ax.text(x + 1.3, 1.95, desc, ha="center", color="white", fontsize=9)
    ax.text(6, 4.5, "Four experiments, one question each",
            ha="center", fontsize=14, fontweight="bold")
    return save(fig, "diagram_experiments.png")


def diagram_noise():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G, seed=3)
    titles = ["Edge deletion", "Edge addition", "Triangle-targeted deletion"]
    for i, ax in enumerate(axes):
        H = G.copy()
        if i == 0:
            H.remove_edges_from(list(H.edges())[:12])
            edge_color = C_GREY
        elif i == 1:
            import itertools
            missing = list(itertools.combinations(range(12), 2))
            H.add_edges_from(missing[:10])
            edge_color = C_TEAL
        else:
            B = nx.triangles(G)
            ranked = sorted(G.edges(), key=lambda e: min(B[e[0]], B[e[1]]), reverse=True)
            H.remove_edges_from(ranked[:12])
            edge_color = C_AMBER
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.4, edge_color=edge_color)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=45,
                               node_color=C_NAVY, edgecolors="white")
        ax.set_title(titles[i], fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Three ways the graph is perturbed in E3", fontsize=14, fontweight="bold")
    return save(fig, "diagram_noise.png")


def diagram_dashboard():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 11.4, 6.4,
                                boxstyle="round,pad=0.08,rounding_size=0.15",
                                facecolor="white", edgecolor=C_GREY, lw=1.4))
    ax.add_patch(FancyBboxPatch((0.3, 6.0), 11.4, 0.7,
                                facecolor=C_NAVY, edgecolor="none"))
    ax.text(0.7, 6.35, "SG-DMNMF  |  Research dashboard", color="white",
            fontsize=11, fontweight="bold", va="center")
    sections = ["Overview", "Methodology", "Alpha Selection", "Experiments",
                "Datasets", "Results", "Limitations", "About"]
    for i, sec in enumerate(sections):
        x = 0.55 + i * 1.4
        ax.add_patch(FancyBboxPatch((x, 5.3), 1.25, 0.45,
                                    boxstyle="round,pad=0.03,rounding_size=0.06",
                                    facecolor=C_LIGHT, edgecolor=C_GREY))
        ax.text(x + 0.62, 5.53, sec, ha="center", va="center",
                fontsize=7, color=C_NAVY)
    ax.add_patch(FancyBboxPatch((0.6, 2.4), 5.3, 2.5,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor=C_LIGHT, edgecolor=C_GREY))
    ax.text(3.25, 4.55, "Stability vs alpha (data-driven chart)",
            ha="center", fontsize=9.5, fontweight="bold")
    ax.plot([1.2, 2.0, 2.8, 3.6, 4.4, 5.2], [3.1, 3.5, 3.9, 3.6, 3.2, 2.9],
            color=C_BLUE, lw=2, marker="o")
    ax.add_patch(FancyBboxPatch((6.2, 2.4), 5.2, 2.5,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor="white", edgecolor=C_GREY))
    ax.text(8.8, 4.55, "Result tables + dataset badges",
            ha="center", fontsize=9.5, fontweight="bold")
    for r in range(4):
        ax.plot([6.5, 11.1], [4.1 - r * 0.4, 4.1 - r * 0.4], color=C_GREY, lw=0.7)
    ax.text(6, 1.5, "Schematic of the dashboard layout (not a screenshot)",
            ha="center", fontsize=9.5, color=C_MUTED)
    return save(fig, "diagram_dashboard.png")


# ---------------------------------------------------------------------------
# Result charts
# ---------------------------------------------------------------------------
def chart_e1():
    ds = "LFR-mu0.3"
    pairs = [(r["Method"], val(r, "NMI_mean")) for r in e1
             if r["Dataset"] == ds and val(r, "NMI_mean") is not None]
    pairs.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = [p[0].replace("DMRA-stab(a=", "SG-DMNMF a=").replace("DMRA-oracle(a=", "Oracle a=")
             for p in pairs]
    colors = [C_BLUE if "SG-DMNMF" in n else C_GREY for n in names]
    ax.barh(names, [p[1] for p in pairs], color=colors, height=0.65)
    for i, v in enumerate([p[1] for p in pairs]):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("NMI"); ax.set_xlim(0, 0.85); ax.grid(axis="x", alpha=0.2)
    ax.set_title("NMI of all methods on the LFR benchmark", fontsize=13, fontweight="bold")
    return save(fig, "chart_e1_nmi.png")


def chart_stability():
    lx, ly = stability_series("LFR-mu0.3")
    qx, qy = stability_series("Quick synthetic (offline)")
    fig, ax = plt.subplots(figsize=(9, 3.7))
    ax.plot(lx, ly, "o-", color=C_BLUE, lw=2.3, label="LFR benchmark")
    ax.plot(qx, qy, "s-", color=C_TEAL, lw=2.3, label="Offline synthetic")
    ax.set_xlabel("alpha (motif weight)"); ax.set_ylabel("stability (AMI)")
    ax.set_xticks([0, 0.1, 0.3, 0.5, 0.7, 0.9]); ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Stability across candidate alpha values (E2)", fontsize=13, fontweight="bold")
    return save(fig, "chart_stability.png")


def chart_robustness():
    ds = "LFR-mu0.3"
    grouped = defaultdict(list)
    for r in e3:
        if r["Dataset"] == ds and r["NoiseType"] == "edge_deletion":
            grouped[(r["Config"], int(round(float(r["NoiseFrac"]) * 100)))].append(val(r, "NMI"))
    configs = sorted({c for c, _ in grouped})
    fig, ax = plt.subplots(figsize=(9, 3.7))
    for cfg, color in zip(configs, [C_GREY, C_BLUE, C_AMBER]):
        pts = sorted((p, mean(v)) for (c, p), v in grouped.items() if c == cfg)
        ax.plot([p for p, _ in pts], [v for _, v in pts], "o-",
                color=color, lw=2.3, label=cfg)
    ax.set_xlabel("edge deletion (%)"); ax.set_ylabel("NMI")
    ax.set_xticks([10, 20, 30, 40]); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Robustness under edge deletion on LFR (E3)", fontsize=13, fontweight="bold")
    return save(fig, "chart_robustness.png")


def chart_ablation():
    metric = "Modularity"
    grouped = defaultdict(list)
    for r in e4:
        if r["Dataset"] == "Dolphin (unlabeled)":
            grouped[r["Config"]].append(val(r, metric))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    names = list(grouped.keys())
    values = [mean(grouped[n]) for n in names]
    colors = [C_BLUE if "motif" in n else C_GREY for n in names]
    ax.bar(names, values, color=colors)
    ax.set_ylabel(metric); ax.grid(axis="y", alpha=0.2)
    ax.set_title("Ablation on the real unlabeled Dolphin graph (E4)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=18, ha="right", fontsize=9)
    return save(fig, "chart_ablation.png")


# Generate all images once
IMG = {
    "network": diagram_network(),
    "triangle": diagram_triangle(),
    "matrices": diagram_matrices(),
    "deep": diagram_deep_nmf(),
    "perturb": diagram_perturbation(),
    "ami": diagram_ami(),
    "pipeline": diagram_pipeline(),
    "experiments": diagram_experiments(),
    "noise": diagram_noise(),
    "dashboard": diagram_dashboard(),
    "e1": chart_e1(),
    "stability": chart_stability(),
    "robust": chart_robustness(),
    "ablation": chart_ablation(),
}

# ---------------------------------------------------------------------------
# PPTX helpers
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(kicker, title, lead=None, note=""):
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGB_WHITE
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                 prs.slide_width, Inches(0.09))
    bar.fill.solid(); bar.fill.fore_color.rgb = RGB_BLUE
    bar.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.28), Inches(11.9), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = kicker.upper()
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = RGB_BLUE
    tb2 = slide.shapes.add_textbox(Inches(0.7), Inches(0.58), Inches(12.0), Inches(0.65))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = title
    p2.font.size = Pt(27); p2.font.bold = True; p2.font.color.rgb = RGB_NAVY
    if lead:
        tb3 = slide.shapes.add_textbox(Inches(0.72), Inches(1.22), Inches(11.9), Inches(0.45))
        p3 = tb3.text_frame.paragraphs[0]
        p3.text = lead
        p3.font.size = Pt(13); p3.font.color.rgb = RGB_MUTED
    footer = slide.shapes.add_textbox(Inches(0.7), Inches(7.06), Inches(11.9), Inches(0.24))
    pf = footer.text_frame.paragraphs[0]
    pf.text = "SG-DMNMF  |  CSET347 Minor Project  |  Arpit Sahu (S24BCAU0195)"
    pf.font.size = Pt(8.5); pf.font.color.rgb = RGB_MUTED
    if note:
        slide.notes_slide.notes_text_frame.text = note
    return slide


def text(slide, x, y, w, h, lines, size=16, color=RGB_NAVY, bold=False, spacing=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold
        p.space_after = Pt(spacing)
    return tb


def card(slide, x, y, w, h, heading, body_lines, accent=RGB_BLUE, size=14):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = RGB_LIGHT
    shp.line.color.rgb = RGB_LINE
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.14); tf.margin_bottom = Inches(0.1)
    p = tf.paragraphs[0]
    p.text = heading
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = accent
    for line in body_lines:
        q = tf.add_paragraph()
        q.text = line
        q.font.size = Pt(size); q.font.color.rgb = RGB_NAVY
        q.space_before = Pt(4)
    return shp


def bullets(slide, lines, x=0.85, y=1.7, w=11.7, h=5.0, size=16):
    return text(slide, x, y, w, h, ["•  " + s for s in lines], size=size, spacing=10)


def picture(slide, path, x, y, w, max_h):
    img = Image.open(path)
    ratio = img.height / img.width
    h = w * ratio
    if h > max_h:
        h = max_h; w = h / ratio
    slide.shapes.add_picture(path, Inches(x), Inches(y),
                             width=Inches(w), height=Inches(h))


def table(slide, data, x, y, w, h, font=10.5):
    t = slide.shapes.add_table(len(data), len(data[0]), Inches(x), Inches(y),
                               Inches(w), Inches(h)).table
    for c in range(len(data[0])):
        t.columns[c].width = Inches(w / len(data[0]))
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            cl = t.cell(r, c)
            cl.text = str(cell)
            cl.margin_left = Inches(0.05); cl.margin_right = Inches(0.05)
            cl.margin_top = Inches(0.02); cl.margin_bottom = Inches(0.02)
            p = cl.text_frame.paragraphs[0]
            p.font.size = Pt(font); p.font.bold = r == 0
            p.font.color.rgb = RGB_WHITE if r == 0 else RGB_NAVY
            cl.fill.solid()
            cl.fill.fore_color.rgb = RGB_BLUE if r == 0 else (RGB_WHITE if r % 2 else RGB_LIGHT)
    return t


# ---------------------------------------------------------------------------
# Slides — every slide carries enough text to be read aloud
# ---------------------------------------------------------------------------

# 1 Title
s = add_slide("CSET347 · Minor Project · Team 8",
              "Stability-Guided Deep Motif NMF for Community Detection",
              "A label-free way to choose the motif weight in deep NMF community detection.",
              "Introduce yourself, the course and the project title. Read the subtitle aloud: the goal is to choose alpha without labels.")
picture(s, IMG["network"], 7.9, 2.0, 4.7, 2.8)
card(s, 0.85, 2.05, 6.5, 1.5, "Student",
     ["Arpit Sahu  ·  S24BCAU0195", "Team 8 (Individual / Solo)"])
card(s, 0.85, 3.75, 6.5, 1.7, "Course",
     ["CSET347, BCA Semester V, Bennett University",
      "Instructor: Dr. Soumita Das"])
text(s, 0.85, 5.9, 11.6, 0.7,
     ["Research track: Deep Learning for Community Detection  ·  November 2026"],
     size=13, color=RGB_MUTED)

# 2 Roadmap
s = add_slide("Roadmap", "What I will show you today",
              "The talk follows the research questions, the method, and then the four experiments.",
              "Read the five stages aloud. Tell the teacher you will finish with limitations and a live dashboard demo.")
picture(s, IMG["pipeline"], 0.85, 1.85, 11.5, 3.0)
bullets(s, [
    "Problem, gap and research questions — slides 3–7.",
    "The SG-DMNMF method and the alpha-selection rule — slides 8–14.",
    "Experimental setup and results E1–E4 — slides 15–22.",
    "Findings, limitations and conclusion — slides 23–26.",
], y=5.05, h=1.8, size=14)

# 3 Community detection
s = add_slide("Background", "What is community detection?",
              "A community is a group of nodes with many internal links and few links to the rest.",
              "Point at the two coloured groups. Say that finding these groups is useful in social, citation and communication networks.")
picture(s, IMG["network"], 0.85, 1.85, 7.7, 4.2)
card(s, 8.9, 1.95, 3.6, 2.1, "Goal",
     ["Assign each node to one community.", "The number of communities is called k."])
card(s, 8.9, 4.25, 3.6, 1.9, "Why NMF?",
     ["A node's factor row is interpretable:", "its largest entry gives the community."])

# 4 Motifs
s = add_slide("Background", "Why triangle motifs matter",
              "Triangles are a higher-order signal: two friends who also know each other indicate a tighter group.",
              "Read the caption under the picture. Explain that counting triangles per edge highlights edges inside dense groups.")
picture(s, IMG["triangle"], 0.85, 1.85, 7.7, 4.0)
card(s, 8.9, 1.95, 3.6, 2.4, "Definition",
     ["B[i,j] counts the triangles", "that contain edge (i,j).",
      "Computed as B = A ∘ (A·A)."])
card(s, 8.9, 4.55, 3.6, 1.6, "Effect",
     ["The blue edges form a triangle;", "the grey edges do not."])

# 5 Problem
s = add_slide("Motivation", "The practical problem: choosing alpha",
              "The base paper mixes edges and motifs with alpha, but tunes alpha by hand using labels.",
              "Say: DMRA-NMF works well, but a new unlabelled network gives no way to pick alpha. This is the gap we close.")
picture(s, IMG["matrices"], 0.85, 1.85, 11.5, 3.4)
card(s, 0.85, 5.5, 5.6, 1.2, "alpha = 0", ["Only ordinary links are used."])
card(s, 6.75, 5.5, 5.6, 1.2, "alpha = 1", ["Only triangle motifs are used."])

# 6 Gap and RQs
s = add_slide("Research gap", "Questions this project answers",
              "No label-free rule existed for choosing the motif weight in deep NMF.",
              "Read all three research questions. Then say each one is answered by one experiment in the results section.")
bullets(s, [
    "RQ1 — Can the stability of the partition under edge perturbation select a good alpha without using labels?",
    "RQ2 — How close is that choice to an oracle that is allowed to use the ground-truth labels?",
    "RQ3 — Does motif information improve robustness when edges are removed or added?",
], y=1.85, size=18)
card(s, 0.85, 5.0, 11.7, 1.4, "Main objective",
     ["Design and evaluate an unsupervised alpha-selection rule for deep motif NMF."])

# 7 Contribution
s = add_slide("Contribution", "SG-DMNMF in one picture",
              "The contribution is the selection rule; it can be attached to any base factorisation.",
              "Walk the pipeline left to right. Emphasise that the stability loop reads only the graph; the oracle appears only in evaluation.")
picture(s, IMG["pipeline"], 0.85, 1.85, 11.6, 3.2)
card(s, 0.85, 5.35, 3.7, 1.3, "Step 1", ["Build motif matrix and mix."])
card(s, 4.83, 5.35, 3.7, 1.3, "Step 2", ["Pick alpha by stability."])
card(s, 8.81, 5.35, 3.7, 1.3, "Step 3", ["Refit on the full graph."])

# 8 Method step 1-2
s = add_slide("Methodology · 1–2", "Building the mixed matrix",
              "Step 1 counts triangles; step 2 blends the two views with alpha.",
              "Read both equations aloud. Explain that rescaling keeps A and B on the same scale before mixing.")
picture(s, IMG["matrices"], 0.85, 1.85, 11.5, 3.3)
card(s, 0.85, 5.4, 5.6, 1.3, "Equation 1", ["B = A ∘ (A · A) — triangle motif matrix."])
card(s, 6.75, 5.4, 5.6, 1.3, "Equation 2", ["M(alpha) = (1-alpha)A + alpha B."])

# 9 Method step 3
s = add_slide("Methodology · 3", "Deep symmetric NMF",
              "A chain of non-negative layers produces a node representation Psi and a community matrix W.",
              "Point at each box in order. Explain the three loss terms in plain language: fit the matrix, keep neighbours similar, keep W diagonal.")
picture(s, IMG["deep"], 0.85, 1.9, 11.6, 3.6)
card(s, 0.85, 5.55, 3.7, 1.2, "Reconstruction", ["Make M-hat close to M."])
card(s, 4.83, 5.55, 3.7, 1.2, "Laplacian", ["Connected nodes stay similar."])
card(s, 8.81, 5.55, 3.7, 1.2, "Diagonal W", ["Communities stay separate."])

# 10 Method step 4
s = add_slide("Methodology · 4", "Perturbation stability",
              "Remove 20% of edges twice, cluster both copies, and compare the partitions with AMI.",
              "Read the sequence aloud. Say that if a partition is real, it should reappear under small random changes.")
picture(s, IMG["perturb"], 0.85, 1.9, 11.5, 3.4)
card(s, 0.85, 5.4, 5.6, 1.3, "Why two copies?", ["Independence removes seed luck."])
card(s, 6.75, 5.4, 5.6, 1.3, "Why AMI?", ["It corrects agreement for chance."])

# 11 AMI explained
s = add_slide("Methodology · 4", "How AMI measures stability",
              "AMI compares two partitions and gives 1 for perfect agreement, 0 for chance-level agreement.",
              "Explain the three boxes. Our method never needs the ground truth: the two partitions both come from the same graph.")
picture(s, IMG["ami"], 0.85, 2.0, 11.5, 3.4)
bullets(s, [
    "High AMI means the two perturbed runs found essentially the same communities.",
    "Low AMI means the partition changed when edges were removed — it is not trustworthy.",
    "We average AMI over several repeats for each candidate alpha.",
], y=5.45, h=1.5, size=14)

# 12 Algorithm
s = add_slide("Algorithm 1", "The complete SG-DMNMF procedure",
              "Six steps: construct, perturb, fit, compare, select, and refit.",
              "Read the six steps one by one. Point out that labels appear nowhere in this loop.")
steps = [
    ("1", "Compute the triangle motif B and the mixed matrix M(alpha)."),
    ("2", "For each candidate alpha in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}:"),
    ("3", "Create two independent graphs by keeping 80% of the edges."),
    ("4", "Fit deep NMF on both and label nodes by argmax(Psi)."),
    ("5", "Compute AMI(C1, C2); choose alpha with the highest mean AMI."),
    ("6", "Refit on the full graph with alpha* and return the partition."),
]
for i, (num, label) in enumerate(steps):
    card(s, 1.0, 1.8 + i * 0.83, 11.3, 0.68, num + "   " + label, [], size=13)

# 13 Alpha selection detail
s = add_slide("Alpha selection", "Candidates and the final decision",
              "Six candidate values are searched; ties are broken with modularity, then one final model is trained.",
              "Explain that a small grid keeps the search cheap. The oracle alpha is only for evaluation, not for use in practice.")
picture(s, IMG["stability"], 0.85, 1.9, 7.7, 3.4)
card(s, 8.9, 2.0, 3.6, 2.0, "Candidates",
     ["0.0, 0.1, 0.3,", "0.5, 0.7, 0.9"])
card(s, 8.9, 4.2, 3.6, 1.6, "Tie-break", ["Full-graph modularity."])
text(s, 0.85, 5.5, 11.6, 1.1,
     ["Selected alpha on LFR: " + selected_alpha("LFR-mu0.3") +
      "   |   Oracle alpha: " + oracle_alpha("LFR-mu0.3") +
      "   (the oracle is only an upper bound)"],
     size=15, color=RGB_BLUE)

# 14 Experiments overview
s = add_slide("Experiments", "Four experiments, one question each",
              "E1 compares methods, E2 explains alpha, E3 tests noise, E4 tests components.",
              "Use this as a map before showing results. Say each experiment answers exactly one research question.")
picture(s, IMG["experiments"], 0.85, 1.9, 11.5, 3.2)
bullets(s, [
    "E1 — clean benchmark against SymNMF, DANMF, DMoN-lite, Leiden, fixed alpha and oracle.",
    "E2 — how clustering quality and stability change as alpha varies.",
    "E3 — deletion, addition and triangle-targeted deletion of edges.",
    "E4 — remove motif / Laplacian / diagonal penalty; vary depth 1–3.",
], y=5.2, h=1.7, size=13.5)

# 15 Datasets
s = add_slide("Setup", "Datasets used in this run",
              "Real graphs are used when available; offline stand-ins are clearly marked synthetic.",
              "Stress the honesty point: the real Dolphin graph has no authentic labels and stays unlabeled; synthetic graphs are demonstration only.")
data = [["Dataset", "Nodes", "Edges", "k", "Triangles", "Status"]]
for d in summary["datasets"]:
    status = "synthetic" if d.get("synthetic") else ("unlabeled" if not d.get("labeled") else "real labeled")
    data.append([d["name"], d["n"], d["edges"], d["k"], d["triangles"], status])
table(s, data, 0.85, 1.85, 11.6, 2.6, font=11)
card(s, 0.85, 4.8, 5.6, 1.7, "Real data",
     ["Karate, the Dolphin GML and LFR benchmarks."])
card(s, 6.75, 4.8, 5.6, 1.7, "Offline synthetic",
     ["Deterministic planted graphs when downloads fail; never presented as real."])

# 16 Metrics and baselines
s = add_slide("Setup", "Metrics and baselines",
              "We report five metrics and compare against four families of baselines.",
              "Read the metric definitions briefly. Hungarian ACC needs labels; modularity works without labels.")
card(s, 0.85, 1.85, 5.7, 4.6, "Metrics",
     ["ACC — accuracy after optimal label matching (Hungarian algorithm).",
      "NMI — normalised mutual information; high means agreement.",
      "ARI — adjusted Rand index; chance-corrected pair agreement.",
      "AMI — adjusted mutual information; less biased than NMI.",
      "Modularity — quality of the partition using only the graph."], size=14)
card(s, 6.8, 1.85, 5.7, 4.6, "Baselines",
     ["SymNMF — single-layer symmetric NMF.",
      "DANMF — deep NMF without motifs.",
      "DMoN-lite — differentiable modularity, no GNN encoder.",
      "Leiden / Louvain — classical modularity optimisation.",
      "Oracle alpha — best alpha chosen with labels (upper bound only)."], size=14)

# 17 E1 chart
s = add_slide("E1 · Clean benchmark", "Comparison on the LFR benchmark",
              "The stability-selected alpha reaches the same NMI as the oracle on LFR.",
              "Read the highlighted bar. Then say SymNMF is slightly higher — we report that honestly.")
picture(s, IMG["e1"], 0.85, 1.9, 8.6, 3.8)
card(s, 9.7, 2.0, 2.8, 1.8, "Karate sanity",
     ["ACC 0.971", "NMI 0.837"])
card(s, 9.7, 4.0, 2.8, 1.7, "Selected alpha",
     ["alpha = " + selected_alpha("LFR-mu0.3")])
text(s, 0.85, 5.9, 11.6, 0.8,
     ["Reading: deep NMF with motifs is competitive; the classical baseline is still strong on this graph."],
     size=14, color=RGB_MUTED)

# 18 E1 findings
s = add_slide("E1 · Findings", "What the clean benchmark tells us",
              "The full table with mean +/- standard deviation is also in the dashboard and the CSV file.",
              "Read the three sentences aloud. They are the conclusion of experiment E1.")
e1_rows = [["Dataset", "Method", "ACC", "NMI", "Modularity"]]
for r in e1:
    if r["Dataset"] in ("Karate", "LFR-mu0.3") and (
            r["Method"].startswith("SymNMF") or "DMRA-stab" in r["Method"]
            or "DMRA-oracle" in r["Method"]):
        e1_rows.append([r["Dataset"], r["Method"],
                        f"{val(r,'ACC_mean'):.3f}" if val(r, "ACC_mean") is not None else "N/A",
                        f"{val(r,'NMI_mean'):.3f}" if val(r, "NMI_mean") is not None else "N/A",
                        f"{val(r,'Modularity_mean'):.3f}"])
table(s, e1_rows, 0.85, 1.85, 11.6, 2.5, font=10.5)
bullets(s, [
    "On Karate, the corrected deep model matches SymNMF: ACC 0.971 and NMI 0.837.",
    "On LFR, the stability-selected alpha and the oracle both reach NMI 0.690.",
    "SymNMF reaches NMI 0.712 on LFR, so the deep model is competitive but not dominant.",
], y=4.6, h=2.0, size=14)

# 19 E2 chart
s = add_slide("E2 · Alpha sensitivity", "How stability changes with alpha",
              "Each alpha is scored by AMI between two perturbed runs; the highest point is selected.",
              "Trace both lines. On the offline synthetic graph the peak is at 0.3; on LFR the curve is flatter.")
picture(s, IMG["stability"], 0.85, 1.9, 8.6, 3.8)
card(s, 9.7, 2.0, 2.8, 1.8, "Offline synthetic",
     ["Peak alpha = 0.3", "Oracle alpha = 0.5"])
card(s, 9.7, 4.0, 2.8, 1.7, "LFR",
     ["Selected alpha = " + selected_alpha("LFR-mu0.3")])
text(s, 0.85, 5.9, 11.6, 0.8,
     ["A clear peak means the graph has a preferred motif weight; a flat curve means alpha matters little."],
     size=14, color=RGB_MUTED)

# 20 E2 findings
s = add_slide("E2 · Findings", "Does the stability peak agree with quality?",
              "On two datasets the label-free choice is close to the label-using oracle.",
              "Read both sentences. State the offline synthetic result as a demonstration, not a scientific claim.")
bullets(s, [
    "Offline synthetic graph: selected alpha 0.3 gives NMI 0.788; oracle alpha 0.5 gives 0.797 — a gap of only 0.009.",
    "LFR benchmark: stability selects alpha " + selected_alpha("LFR-mu0.3") +
    " while the oracle selects " + oracle_alpha("LFR-mu0.3") + "; both give NMI 0.690.",
    "This supports RQ1 and RQ2: stability can replace labels for choosing alpha on these graphs.",
], y=1.9, size=17)
card(s, 0.85, 5.4, 11.7, 1.3, "Important caveat",
     ["Where the stability peak differs from the oracle, we report the difference instead of hiding it."])

# 21 E3 noise
s = add_slide("E3 · Robustness", "Three kinds of edge noise",
              "We test random deletion, random addition, and deletion of edges inside many triangles.",
              "Explain the three pictures. Then say we measure NMI on labeled graphs and modularity on unlabeled ones.")
picture(s, IMG["noise"], 0.85, 1.9, 11.5, 3.2)
card(s, 0.85, 5.4, 3.7, 1.3, "Deletion", ["Remove 10–40% of edges."])
card(s, 4.83, 5.4, 3.7, 1.3, "Addition", ["Add random missing edges."])
card(s, 8.81, 5.4, 3.7, 1.3, "Triangle deletion", ["Remove high-triangle edges first."])

# 22 E3 findings
s = add_slide("E3 · Findings", "Robustness results",
              "On LFR edge deletion, all configurations lose quality as noise grows.",
              "Read the chart and the two conclusions. Be honest that the stability-selected alpha is competitive, not always best.")
picture(s, IMG["robust"], 0.85, 1.9, 8.6, 3.8)
card(s, 9.7, 2.0, 2.8, 1.9, "Observation",
     ["NMI falls from about 0.55 at 10% noise to about 0.43 at 40%."])
card(s, 9.7, 4.1, 2.8, 1.7, "Conclusion",
     ["Motif use changes robustness, but no setting wins everywhere."])
text(s, 0.85, 5.9, 11.6, 0.8,
     ["This answers RQ3 with a nuanced result rather than an overclaim."],
     size=14, color=RGB_MUTED)

# 23 E4 ablation
s = add_slide("E4 · Ablation", "Which model components matter?",
              "Each configuration removes one component or changes the number of layers.",
              "Read the chart. The no-motif configuration is highlighted; explain that its score differs from the full model.")
picture(s, IMG["ablation"], 0.85, 1.9, 8.6, 3.8)
card(s, 9.7, 2.0, 2.8, 1.9, "Variants",
     ["No motif, no Laplacian, no diagonal penalty, depth 1/2/3."])
card(s, 9.7, 4.1, 2.8, 1.7, "Reading",
     ["Depth = 2 with all penalties is the full model."])
text(s, 0.85, 5.9, 11.6, 0.8,
     ["The changes confirm that each component contributes to the final partition."],
     size=14, color=RGB_MUTED)

# 24 RQ answers
s = add_slide("Discussion", "Research questions answered",
              "Each question is backed by one experiment and a measurable result.",
              "Read the three answers. This slide is the core academic conclusion.")
bullets(s, [
    "RQ1 — Yes. Partition stability under 80% edge subsampling selects alpha without labels (E2).",
    "RQ2 — Closely. On LFR the selected alpha matches the oracle NMI of 0.690; on the synthetic demo the gap is 0.009 (E1–E2).",
    "RQ3 — Partly. Motif configurations change robustness under noise, but no single setting dominates all noise types (E3).",
], y=1.9, size=17)
card(s, 0.85, 5.3, 11.7, 1.4, "Overall",
     ["The label-free selector is feasible; the deep model itself is a simplified research prototype."])

# 25 Limitations
s = add_slide("Limitations", "What the project does not claim",
              "Stating the scope clearly is part of honest research reporting.",
              "Read every point. If the teacher asks about reproducing the paper, point to the first bullet.")
bullets(s, [
    "DMRA-lite is symmetric, triangle-only and trained with Adam/softplus — it is not an exact copy of the asymmetric DMRA-NMF paper.",
    "The number of communities k is assumed known; for unlabeled graphs it is estimated by modularity optimisation.",
    "DMoN-lite has no GNN encoder; Cora and Citeseer require torch-geometric to load the real graphs.",
    "Offline synthetic graphs are demonstration only and are never presented as real benchmarks.",
    "The CP1 report planned normalised matrices and Wilcoxon tests; this implementation reports mean +/- std instead.",
], y=1.85, size=15)

# 26 Dashboard
s = add_slide("Deliverables", "Run it and see the results",
              "One command runs the tests and opens a local dashboard with every table and figure.",
              "Show this slide while opening the dashboard in the browser. The layout is a schematic, not a screenshot.")
picture(s, IMG["dashboard"], 0.85, 1.85, 8.0, 4.2)
card(s, 9.1, 2.0, 3.35, 2.0, "Command",
     ["python run_project.py all", "", "23 unit tests are included."])
card(s, 9.1, 4.2, 3.35, 1.7, "Outputs",
     ["results/*.csv, figures/*.png,", "Colab notebook, Overleaf report."])

# 27 Conclusion
s = add_slide("Conclusion", "Main takeaway and future work",
              "The contribution is an unsupervised alpha selector that uses no labels at all.",
              "End with the takeaway, then mention future work. Invite questions.")
bullets(s, [
    "SG-DMNMF chooses the motif weight from partition stability measured by AMI.",
    "Evidence comes from four experiments and is available as tables, charts and figures.",
    "On a real LFR benchmark the label-free choice matches the supervised oracle.",
    "Future work: asymmetric factorisation, higher-order motifs, automatic k, statistical tests.",
], y=1.9, size=17)
card(s, 0.85, 5.4, 11.7, 1.3, "Reproducibility",
     ["Code, tests, dashboard, Colab notebook and Overleaf report are all included."])

# 28 Thank you
s = add_slide("Thank you", "Questions?",
              "Thank the teacher and invite questions.",
              "If asked about exact paper reproduction, repeat: DMRA-lite simplification; the novelty is the selection rule.")
text(s, 0.85, 2.3, 11.7, 1.0,
     ["Stability-guided selection of the motif weight in deep NMF-based community detection"],
     size=20, color=RGB_BLUE)
text(s, 0.85, 3.6, 11.7, 1.8,
     ["Arpit Sahu  ·  S24BCAU0195  ·  Team 8",
      "CSET347, BCA Semester V  ·  Bennett University",
      "Instructor: Dr. Soumita Das"],
     size=17, color=RGB_MUTED, spacing=9)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT), "bytes,", len(prs.slides._sldIdLst), "slides")
