#!/usr/bin/env python3
"""Generate all figures for the Neuron Suppression Awareness write-up."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# ── Anthropic-inspired palette ─────────────────────────────────────────
CORAL = "#E07A5F"
TEAL = "#457B9D"
SLATE = "#2B2D42"
WARM_BG = "#F4F1DE"
GOLD = "#E9C46A"
CORAL_LIGHT = "#F2B8A8"
TEAL_LIGHT = "#A8D0E0"
WHITE = "#FFFFFF"

# ── Global theme ───────────────────────────────────────────────────────
def apply_theme():
    plt.rcParams.update({
        "font.family": "Helvetica Neue",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": SLATE,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.6,
        "axes.facecolor": WHITE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": WHITE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 11,
        "text.color": SLATE,
    })

apply_theme()


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 1 — Attack Effectiveness (Phase 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig1_attack_effectiveness():
    with open(ROOT / "results/phase1/phase1_results.json") as f:
        d = json.load(f)
    clean, supp = d["metrics"]["clean_asr"], d["metrics"]["suppressed_asr"]

    fig, ax = plt.subplots(figsize=(5, 5.5))

    bars = ax.bar(
        [0, 1], [clean, supp],
        width=0.55,
        color=[TEAL, CORAL],
        edgecolor=[TEAL, CORAL],
        linewidth=0.8,
        zorder=3,
    )

    for bar, val in zip(bars, [clean, supp]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{val:.0%}",
            ha="center", va="bottom",
            fontsize=18, fontweight="bold", color=SLATE,
        )

    ax.annotate(
        "", xy=(1, supp - 0.01), xytext=(0, clean + 0.06),
        arrowprops=dict(
            arrowstyle="->,head_width=0.3,head_length=0.15",
            color=SLATE, lw=1.2,
            connectionstyle="arc3,rad=-0.15",
        ),
    )
    ax.text(
        0.5, 0.55, "+93 pp",
        ha="center", va="center",
        fontsize=12, fontstyle="italic", color=SLATE,
        transform=ax.transData,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean", "Neuron\nSuppressed"], fontsize=12)
    ax.set_ylabel("Attack Success Rate", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("Neuron Suppression Attack Effectiveness", pad=14)

    fig.text(
        0.5, -0.02,
        "Pinning one MLP neuron (L14:N7924 = +20) raises ASR\n"
        "from 5% to 98% on 100 JailbreakBench prompts.",
        ha="center", va="top", fontsize=9.5, color="#666666",
        fontstyle="italic",
    )

    fig.tight_layout()
    save(fig, "fig1_attack_effectiveness")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 2 — The Detection Gap (Phase 3A vs Phase 5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig2_detection_gap():
    phase3 = {
        "Clean\nFPR": 0.00,
        "CAA\nDetection": 1.00,
        "Noise\nFPR": 0.00,
        "Suppression\nDetection": 0.00,
    }
    with open(ROOT / "phase5-output/artifacts/phase5/20260531T141944Z/phase5_results.json") as f:
        p5 = json.load(f)
    phase5 = {
        "Clean\nFPR": p5["clean_fpr"],
        "CAA\nDetection": p5["caa_detection_rate"],
        "Noise\nFPR": p5["noise_fpr"],
        "Suppression\nDetection": p5["suppression_detection_rate"],
    }

    labels = list(phase3.keys())
    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars3 = ax.bar(
        x - w / 2, list(phase3.values()), w,
        label="CAA-Only Training (Phase 3)",
        color="none", edgecolor=TEAL, linewidth=2.0,
        hatch="///", zorder=3,
    )
    bars5 = ax.bar(
        x + w / 2, list(phase5.values()), w,
        label="Mixed Training (Phase 5)",
        color=TEAL, edgecolor=TEAL, linewidth=0.8,
        zorder=3,
    )

    bars3[3].set_edgecolor(CORAL)
    bars5[3].set_color(CORAL)
    bars5[3].set_edgecolor(CORAL)

    for bars in (bars3, bars5):
        for bar, val in zip(bars, bars.datavalues):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f"{val:.0%}",
                    ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=SLATE,
                )

    ax.annotate(
        "0% to 100%",
        xy=(x[3] - w / 2, 0.02),
        xytext=(x[3] - w / 2, 0.40),
        ha="center", fontsize=10, fontweight="bold", color=CORAL,
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=11)
    ax.set_title("Detection Transfer: CAA-Only vs Mixed Training", pad=28)

    fig.text(
        0.5, -0.02,
        "CAA-only training: perfect CAA detection but 0% suppression detection.\n"
        "Mixed training: perfect detection of both with zero false positives.",
        ha="center", va="top", fontsize=9.5, color="#666666",
        fontstyle="italic",
    )

    fig.tight_layout()
    save(fig, "fig2_detection_gap")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 3 — Susceptibility Paradox (Phase 3B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig3_susceptibility_paradox():
    data = np.array([
        [0.00, 0.96],
        [0.20, 0.96],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("teal_coral", [TEAL, WARM_BG, CORAL])

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Suppression", "With Suppression"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Base Model", "Steering-Aware\nModel"], fontsize=12)
    ax.tick_params(length=0)

    for i in range(2):
        for j in range(2):
            val = data[i, j]
            color = WHITE if val > 0.6 else SLATE
            ax.text(
                j, i, f"{val:.0%}",
                ha="center", va="center",
                fontsize=24, fontweight="bold", color=color,
            )

    ax.set_title("Susceptibility Paradox: Attack Success Rate", pad=14)

    fig.text(
        0.5, -0.04,
        "Detection training provides zero protection (both reach 96% ASR)\n"
        "while degrading baseline safety from 0% to 20%.",
        ha="center", va="top", fontsize=9.5, color="#666666",
        fontstyle="italic",
    )

    fig.tight_layout()
    save(fig, "fig3_susceptibility_paradox")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 4 — Subspace Geometry (Phase 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig4_subspace_geometry():
    layers = [14, 18, 24, 30]
    supp_l2 = [35.99, 43.09, 103.00, 238.40]
    caa_l2 = [0.00, 0.00, 467.99, 681.62]
    cosines = [0.00, 0.00, 0.0149, 0.1673]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: L2 norms
    x = np.arange(len(layers))
    w = 0.32
    ax1.bar(x - w / 2, supp_l2, w, label="Suppression", color=CORAL, edgecolor=CORAL, zorder=3)
    ax1.bar(x + w / 2, caa_l2, w, label="CAA", color=TEAL, edgecolor=TEAL, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{l}" for l in layers], fontsize=11)
    ax1.set_ylabel("Perturbation L2 Norm", fontsize=12)
    ax1.yaxis.grid(True, zorder=0)
    ax1.legend(fontsize=10)
    ax1.set_title("A.  Perturbation Magnitude", pad=10, loc="left")

    ax1.annotate(
        "detector\nlayer",
        xy=(2, max(supp_l2[2], caa_l2[2]) + 20),
        xytext=(2, caa_l2[2] + 120),
        ha="center", fontsize=9, color=SLATE,
        arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.0),
    )

    ax1.text(
        2 - w / 2, supp_l2[2] + 15, "22%",
        ha="center", fontsize=9, fontweight="bold", color=CORAL,
    )

    # Panel B: Cosine similarity
    ax2.plot(x, cosines, "o-", color=SLATE, lw=2, markersize=8, zorder=3)
    ax2.axhline(0, color="#CCCCCC", lw=1, ls="--", zorder=1)

    for xi, c in zip(x, cosines):
        label = f"{c:.3f}" if c > 0 else "0"
        offset = 0.025 if c < 0.1 else -0.025
        ax2.text(
            xi, c + offset, label,
            ha="center", va="bottom" if c < 0.1 else "top",
            fontsize=10, fontweight="bold", color=SLATE,
        )

    ax2.axvspan(1.5, 2.5, alpha=0.08, color=GOLD, zorder=0)
    ax2.text(2, -0.06, "detector layer", ha="center", fontsize=9, color=SLATE, fontstyle="italic")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{l}" for l in layers], fontsize=11)
    ax2.set_ylabel("Cosine Similarity\n(Suppression vs CAA)", fontsize=12)
    ax2.set_ylim(-0.1, 0.3)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_title("B.  Directional Alignment", pad=10, loc="left")

    fig.text(
        0.5, -0.02,
        "At layer 24, suppression is ~22% the magnitude of CAA and nearly orthogonal (cos = 0.015).\n"
        "The detector monitors a subspace that suppression does not occupy.",
        ha="center", va="top", fontsize=9.5, color="#666666",
        fontstyle="italic",
    )

    fig.tight_layout(w_pad=3)
    save(fig, "fig4_subspace_geometry")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 5 — Readable but Not Self-Reported (Phase 4 Probes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig5_readable_not_reported():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5), gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: External probe vs self-report
    categories = ["CAA", "Suppression"]
    probe_acc = [1.0, 1.0]
    self_report = [1.0, 0.0]

    x = np.arange(len(categories))
    w = 0.30

    b1 = ax1.bar(x - w / 2, probe_acc, w, label="External Probe", color=TEAL, edgecolor=TEAL, zorder=3)
    b2 = ax1.bar(x + w / 2, self_report, w, label="Self-Report", color=[TEAL, CORAL], edgecolor=[TEAL, CORAL], zorder=3)

    for bar, val in zip(b1, probe_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.0%}",
                 ha="center", fontsize=13, fontweight="bold", color=SLATE)
    for bar, val in zip(b2, self_report):
        y = max(val, 0.02)
        ax1.text(bar.get_x() + bar.get_width() / 2, y + 0.025, f"{val:.0%}",
                 ha="center", fontsize=13, fontweight="bold",
                 color=CORAL if val == 0 else SLATE)

    ax1.annotate(
        "readable but\nnot self-reported",
        xy=(1 + w / 2, 0.03),
        xytext=(1.4, 0.45),
        ha="center", fontsize=10, color=CORAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.set_ylabel("Accuracy / Detection Rate", fontsize=12)
    ax1.set_ylim(0, 1.22)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.yaxis.grid(True, zorder=0)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("A.  External vs Self-Report Detection", pad=10, loc="left")

    # Panel B: Probe direction cosine alignment
    comparisons = ["Suppression\nDelta", "CAA\nDelta"]
    cosines = [0.4466, 0.0084]
    colors = [CORAL, TEAL]

    bars = ax2.barh([0, 1], cosines, height=0.45, color=colors, edgecolor=colors, zorder=3)
    for bar, val in zip(bars, cosines):
        ax2.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=12, fontweight="bold", color=SLATE)

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(comparisons, fontsize=11)
    ax2.set_xlabel("Cosine with Probe Direction", fontsize=12)
    ax2.set_xlim(0, 0.6)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_title("B.  Probe Direction Alignment", pad=10, loc="left")
    ax2.invert_yaxis()

    fig.text(
        0.5, -0.02,
        "External probes perfectly classify suppression, but the model's self-report is blind to it.\n"
        "The probe direction aligns with suppression (cos = 0.45) but not CAA (cos = 0.008).",
        ha="center", va="top", fontsize=9.5, color="#666666",
        fontstyle="italic",
    )

    fig.tight_layout(w_pad=3)
    save(fig, "fig5_readable_not_reported")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 6 — Training Dynamics (Phase 2B + Phase 5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig6_training_dynamics():
    phase2b_loss = [0.2167, 0.1270, 0.0880, 0.0648]

    with open(ROOT / "phase5-output/artifacts/phase5/20260531T141944Z/train_metrics.jsonl") as f:
        phase5_loss = [json.loads(line)["mean_loss"] for line in f]

    epochs = [1, 2, 3, 4]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(epochs, phase2b_loss, "s--", color=TEAL, lw=2.2, markersize=8,
            label="Phase 2B: CAA-Only", zorder=3)
    ax.plot(epochs, phase5_loss, "o-", color=CORAL, lw=2.2, markersize=8,
            label="Phase 5: Mixed (CAA + Suppression)", zorder=3)

    for ep, v2, v5 in zip(epochs, phase2b_loss, phase5_loss):
        if ep == 4:
            ax.text(ep + 0.1, v2 + 0.003, f"{v2:.3f}", fontsize=9, color=TEAL, va="bottom")
            ax.text(ep + 0.1, v5 - 0.003, f"{v5:.3f}", fontsize=9, color=CORAL, va="top")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Training Loss", fontsize=12)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.25)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("Training Convergence", pad=14)

    fig.text(
        0.5, -0.02,
        "Both runs converge smoothly. Mixed training achieves lower final loss (0.045 vs 0.065)\n"
        "despite the broader detection task.",
        ha="center", va="top", fontsize=9.5, color="#666666",
        fontstyle="italic",
    )

    fig.tight_layout()
    save(fig, "fig6_training_dynamics")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run all
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("Generating figures...\n")
    fig1_attack_effectiveness()
    fig2_detection_gap()
    fig3_susceptibility_paradox()
    fig4_subspace_geometry()
    fig5_readable_not_reported()
    fig6_training_dynamics()
    print("\nDone! All figures saved to:", OUT)
