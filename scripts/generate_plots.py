"""Generate publication quality evaluation plots from metrics_summary.json.

Reads strictly from docs/evaluation/metrics_summary.json and outputs up to 7 PNG
figures into docs/evaluation/figures/ using the non-interactive matplotlib Agg backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "evaluation"
FIGURES_DIR = EVAL_DIR / "figures"
METRICS_PATH = EVAL_DIR / "metrics_summary.json"
DPI = 200

MODEL_ORDER = [
    "pairwise_lgbm",
    "sublinear_tfidf",
    "raw_tfidf",
    "markov_jsd",
    "markov_frobenius",
]

MODEL_META: dict[str, dict[str, Any]] = {
    "pairwise_lgbm": {
        "label": "Pairwise LightGBM (Markov + AST)",
        "short": "Pairwise\nLightGBM",
        "color": "#1b4965",
        "linestyle": "-",
        "linewidth": 2.6,
        "marker": "o",
        "zorder": 5,
    },
    "sublinear_tfidf": {
        "label": "Sublinear TF-IDF Cosine",
        "short": "Sublinear\nTF-IDF",
        "color": "#d95f02",
        "linestyle": "--",
        "linewidth": 1.9,
        "marker": "s",
        "zorder": 4,
    },
    "raw_tfidf": {
        "label": "Raw TF-IDF Cosine",
        "short": "Raw\nTF-IDF",
        "color": "#e6ab02",
        "linestyle": "-.",
        "linewidth": 1.8,
        "marker": "^",
        "zorder": 3,
    },
    "markov_jsd": {
        "label": "Markov JSD (1 - JSD)",
        "short": "Markov\nJSD",
        "color": "#7570b3",
        "linestyle": ":",
        "linewidth": 2.0,
        "marker": "D",
        "zorder": 2,
    },
    "markov_frobenius": {
        "label": "Markov Frobenius (-d_F)",
        "short": "Markov\nFrobenius",
        "color": "#66a61e",
        "linestyle": (0, (3, 1, 1, 1)),
        "linewidth": 1.8,
        "marker": "v",
        "zorder": 1,
    },
}


def _configure_style() -> None:
    """Apply clean publication styling for all figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _compute_average_precision(precision: list[float], recall: list[float]) -> float:
    """Compute step-function Average Precision from sklearn precision_recall_curve arrays."""
    p = np.asarray(precision, dtype=float)
    r = np.asarray(recall, dtype=float)
    return float(-np.sum(np.diff(r) * p[:-1]))


def plot_roc_curves(data: dict[str, Any], out_path: Path) -> None:
    """Plot ConPlag v2 5-fold OOF ROC curves for all 5 models."""
    cv = data["conplag_v2_cv"]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    for model_key in MODEL_ORDER:
        entry = cv[model_key]
        meta = MODEL_META[model_key]
        fpr = entry["roc_curve"]["fpr"]
        tpr = entry["roc_curve"]["tpr"]
        auc_val = entry["micro_oof"]["roc_auc"]
        ax.plot(
            fpr,
            tpr,
            label=f"{meta['label']} (AUC = {auc_val:.4f})",
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=meta["linewidth"],
            zorder=meta["zorder"],
        )

    # Highlight Pairwise LightGBM operating points
    lgbm_oof = cv["pairwise_lgbm"]["micro_oof"]
    lgbm_low = cv["pairwise_lgbm"]["micro_oof_low_fpr"]
    ax.scatter(
        [lgbm_oof["fpr"]],
        [lgbm_oof["recall"]],
        color="#1b4965",
        edgecolor="white",
        s=70,
        zorder=6,
        label=(
            f"LightGBM Balanced (FPR={lgbm_oof['fpr']:.3f}, "
            f"TPR={lgbm_oof['recall']:.3f})"
        ),
    )
    ax.scatter(
        [lgbm_low["fpr"]],
        [lgbm_low["recall"]],
        color="#c1121f",
        edgecolor="white",
        marker="D",
        s=65,
        zorder=6,
        label=(
            f"LightGBM Low-FPR (FPR={lgbm_low['fpr']:.3f}, "
            f"TPR={lgbm_low['recall']:.3f})"
        ),
    )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label="Random Chance (AUC = 0.5000)",
        zorder=0,
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ConPlag v2 5-Fold Problem-Disjoint OOF ROC Curves")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(data: dict[str, Any], out_path: Path) -> None:
    """Plot ConPlag v2 5-fold OOF Precision-Recall curves for all 5 models."""
    cv = data["conplag_v2_cv"]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    # Derive positive class prevalence directly from confusion matrix
    cm0 = cv["pairwise_lgbm"]["micro_oof"]["confusion_matrix"]
    total_pos = cm0[1][0] + cm0[1][1]
    total_all = cm0[0][0] + cm0[0][1] + total_pos
    prevalence = total_pos / total_all

    for model_key in MODEL_ORDER:
        entry = cv[model_key]
        meta = MODEL_META[model_key]
        prec = entry["pr_curve"]["precision"]
        rec = entry["pr_curve"]["recall"]
        f1_val = entry["micro_oof"]["f1"]
        ap_val = _compute_average_precision(prec, rec)
        ax.plot(
            rec,
            prec,
            label=f"{meta['label']} (AP = {ap_val:.4f}, F1 = {f1_val:.4f})",
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=meta["linewidth"],
            zorder=meta["zorder"],
        )

    lgbm_oof = cv["pairwise_lgbm"]["micro_oof"]
    ax.scatter(
        [lgbm_oof["recall"]],
        [lgbm_oof["precision"]],
        color="#1b4965",
        edgecolor="white",
        s=70,
        zorder=6,
        label=(
            f"LightGBM Balanced (P={lgbm_oof['precision']:.3f}, "
            f"R={lgbm_oof['recall']:.3f})"
        ),
    )

    ax.axhline(
        prevalence,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label=f"No-Skill Prevalence ({prevalence:.4f})",
        zorder=0,
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("ConPlag v2 5-Fold Problem-Disjoint OOF Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=8.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ir_plag_levels(data: dict[str, Any], out_path: Path) -> None:
    """Plot Zero-Shot IR-Plag per-level recall (L1-L6) and NON_PLAG FPR."""
    ir = data["ir_plag_zero_shot"]
    levels = ["L1", "L2", "L3", "L4", "L5", "L6"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # Left panel: Pairwise LightGBM Balanced vs Low-FPR across L1-L6 + NON_PLAG FPR
    lgbm = ir["pairwise_lgbm"]
    bal_vals = [lgbm["per_level_at_threshold"][lvl] for lvl in levels]
    low_vals = [lgbm["per_level_at_threshold_low_fpr"][lvl] for lvl in levels]
    bal_fpr = lgbm["per_level_at_threshold"]["NON_PLAG_fpr"]
    low_fpr = lgbm["per_level_at_threshold_low_fpr"]["NON_PLAG_fpr"]

    categories = levels + ["NON_PLAG\n(FPR)"]
    bal_series = bal_vals + [bal_fpr]
    low_series = low_vals + [low_fpr]

    x = np.arange(len(categories))
    width = 0.34
    offset = width / 2 + 0.025

    bal_thr = lgbm["at_threshold"]["threshold"]
    low_thr = lgbm["at_threshold_low_fpr"]["threshold"]

    bars_bal = ax_left.bar(
        x - offset,
        bal_series,
        width,
        label=f"Balanced (thr = {bal_thr:.4f})",
        color="#1b4965",
        edgecolor="#102a3c",
        linewidth=0.7,
    )
    bars_low = ax_left.bar(
        x + offset,
        low_series,
        width,
        label=f"Low-FPR (thr = {low_thr:.4f})",
        color="#62b6cb",
        edgecolor="#1b4965",
        linewidth=0.7,
    )

    # Visually distinguish the NON_PLAG FPR bars
    bars_bal[-1].set_facecolor("#c1121f")
    bars_bal[-1].set_edgecolor("#780000")
    bars_bal[-1].set_hatch("//")
    bars_low[-1].set_facecolor("#f4a261")
    bars_low[-1].set_edgecolor("#9e4700")
    bars_low[-1].set_hatch("//")

    ax_left.axvline(5.5, color="#555555", linestyle="--", linewidth=1.0)
    ax_left.text(
        6.0,
        0.82,
        "FPR\n(Lower is\nBetter)",
        fontsize=8.2,
        color="#780000",
        ha="center",
        va="center",
        fontweight="bold",
    )

    for bar in list(bars_bal) + list(bars_low):
        h = bar.get_height()
        label_txt = "100%" if h >= 0.9995 else f"{h * 100:.1f}%"
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.018,
            label_txt,
            ha="center",
            va="bottom",
            fontsize=6.6,
            rotation=0,
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(categories)
    ax_left.set_ylim(0.0, 1.18)
    ax_left.set_ylabel("Rate (Recall for L1-L6; FPR for NON_PLAG)")
    ax_left.set_title("Pairwise LightGBM Zero-Shot IR-Plag: Balanced vs Low-FPR")
    ax_left.legend(loc="upper right", fontsize=8.3)

    # Right panel: All 5 models per-level recall profile at Balanced threshold
    x_lvl = np.arange(len(levels))
    for model_key in MODEL_ORDER:
        meta = MODEL_META[model_key]
        per_lvl = ir[model_key]["per_level_at_threshold"]
        recalls = [per_lvl[lvl] for lvl in levels]
        fpr_val = per_lvl["NON_PLAG_fpr"]
        ax_right.plot(
            x_lvl,
            recalls,
            label=f"{meta['label']} (FPR={fpr_val * 100:.1f}%)",
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=meta["linewidth"],
            marker=meta["marker"],
            markersize=5.5,
            zorder=meta["zorder"],
        )

    ax_right.set_xticks(x_lvl)
    ax_right.set_xticklabels(levels)
    ax_right.set_ylim(0.0, 1.08)
    ax_right.set_xlabel("Faidhi & Robinson Plagiarism Obfuscation Level")
    ax_right.set_ylabel("Zero-Shot Recall")
    ax_right.set_title("All Models Per-Level Recall (Balanced Threshold)")
    ax_right.legend(loc="lower left", fontsize=8.0)

    fig.suptitle(
        "IR-Plag External Zero-Shot Evaluation (Trained on ConPlag v2, strip_boilerplate=True)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_boilerplate_ablation(data: dict[str, Any], out_path: Path) -> None:
    """Plot ConPlag v2 vs v1 F1 ablation for strip_boilerplate=True vs False."""
    cross = data["conplag_v1_cross_domain"]
    strip_true = cross["strip_boilerplate_true"]
    strip_false = cross["strip_boilerplate_false"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # Left panel: Pairwise LightGBM v2 OOF F1 vs v1 Zero-Shot F1
    lgbm_t = strip_true["pairwise_lgbm"]
    lgbm_f = strip_false["pairwise_lgbm"]

    regimes = ["strip_boilerplate = True\n(Production)", "strip_boilerplate = False\n(Unstripped)"]
    v2_f1 = [lgbm_t["v2_oof_f1"], lgbm_f["v2_oof_f1"]]
    v1_f1 = [lgbm_t["metrics"]["f1"], lgbm_f["metrics"]["f1"]]
    shifts = [lgbm_t["f1_shift"], lgbm_f["f1_shift"]]

    x = np.arange(len(regimes))
    width = 0.34

    bars_v2 = ax_left.bar(
        x - width / 2,
        v2_f1,
        width,
        label="ConPlag v2 (5-Fold OOF F1)",
        color="#1b4965",
        edgecolor="#102a3c",
        linewidth=0.7,
    )
    bars_v1 = ax_left.bar(
        x + width / 2,
        v1_f1,
        width,
        label="ConPlag v1 (5-Fold OOF F1)",
        color="#62b6cb",
        edgecolor="#1b4965",
        linewidth=0.7,
    )

    for bar in list(bars_v2) + list(bars_v1):
        h = bar.get_height()
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )

    best_abs_shift = min(abs(s) for s in shifts)
    for idx, shift_val in enumerate(shifts):
        top_y = max(v2_f1[idx], v1_f1[idx]) + 0.06
        badge_color = "#2a9d8f" if abs(shift_val) <= best_abs_shift + 1e-12 else "#c1121f"
        ax_left.annotate(
            f"Shift: {shift_val * 100:+.2f}%",
            xy=(x[idx], top_y),
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": badge_color,
                "edgecolor": "none",
            },
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(regimes)
    ax_left.set_ylim(0.0, 0.96)
    ax_left.set_ylabel("F1-Score")
    ax_left.set_title("Pairwise LightGBM: ConPlag v2 vs v1 Boilerplate Ablation")
    ax_left.legend(loc="lower right", fontsize=8.5)

    # Right panel: All 5 models F1 shift comparison (strip=True vs strip=False)
    x_m = np.arange(len(MODEL_ORDER))
    shifts_t = [strip_true[m]["f1_shift"] * 100.0 for m in MODEL_ORDER]
    shifts_f = [strip_false[m]["f1_shift"] * 100.0 for m in MODEL_ORDER]
    short_labels = [MODEL_META[m]["short"] for m in MODEL_ORDER]

    w2 = 0.36
    b_t = ax_right.bar(
        x_m - w2 / 2,
        shifts_t,
        w2,
        label="strip_boilerplate = True",
        color="#2a9d8f",
        edgecolor="#1d6e64",
        linewidth=0.7,
    )
    b_f = ax_right.bar(
        x_m + w2 / 2,
        shifts_f,
        w2,
        label="strip_boilerplate = False",
        color="#e76f51",
        edgecolor="#9e3d24",
        linewidth=0.7,
    )

    ax_right.axhline(0.0, color="#333333", linewidth=0.9)
    for bar in list(b_t) + list(b_f):
        h = bar.get_height()
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            h - 0.35,
            f"{h:+.2f}%",
            ha="center",
            va="top",
            fontsize=7.8,
            fontweight="bold",
        )

    ax_right.set_xticks(x_m)
    ax_right.set_xticklabels(short_labels)
    min_shift = min(shifts_t + shifts_f)
    ax_right.set_ylim(min(min_shift - 2.2, -13.5), 1.2)
    ax_right.set_ylabel("Cross-Domain F1 Shift: (v1 - v2) in Percentage Points")
    ax_right.set_title("Cross-Domain F1 Drop Across All Models (Closer to 0% is Better)")
    ax_right.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(data: dict[str, Any], out_path: Path) -> None:
    """Plot side-by-side confusion matrices for Pairwise LightGBM across regimes."""
    lgbm_v2_bal = data["conplag_v2_cv"]["pairwise_lgbm"]["micro_oof"]
    lgbm_v2_low = data["conplag_v2_cv"]["pairwise_lgbm"]["micro_oof_low_fpr"]
    lgbm_v1 = data["conplag_v1_cross_domain"]["strip_boilerplate_true"]["pairwise_lgbm"]["metrics"]
    lgbm_ir = data["ir_plag_zero_shot"]["pairwise_lgbm"]["at_threshold"]

    panels = [
        (f"ConPlag v2 OOF\n(Balanced, thr={lgbm_v2_bal['threshold']:.4f})", lgbm_v2_bal),
        (f"ConPlag v2 OOF\n(Low-FPR, thr={lgbm_v2_low['threshold']:.4f})", lgbm_v2_low),
        (f"ConPlag v1 5-Fold OOF\n(strip=True, thr={lgbm_v1['threshold']:.4f})", lgbm_v1),
        (f"IR-Plag Zero-Shot\n(Balanced, thr={lgbm_ir['threshold']:.4f})", lgbm_ir),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3))
    class_labels = ["Non-Plag (0)", "Plag (1)"]

    for ax, (title, metrics) in zip(axes, panels, strict=True):
        cm = np.asarray(metrics["confusion_matrix"], dtype=int)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

        ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels, fontsize=8.8)
        ax.set_yticklabels(class_labels, fontsize=8.8)
        ax.set_xlabel("Predicted Label", fontsize=9.2)
        if ax is axes[0]:
            ax.set_ylabel("True Label", fontsize=9.2)
        ax.grid(False)

        for row in range(2):
            for col in range(2):
                val = cm[row, col]
                pct = cm_pct[row, col] * 100.0
                text_color = "white" if cm_pct[row, col] > 0.55 else "#111111"
                ax.text(
                    col,
                    row,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10.5,
                    fontweight="bold",
                )

        subtitle = (
            f"F1={metrics['f1']:.3f} | P={metrics['precision']:.3f}\n"
            f"R={metrics['recall']:.3f} | FPR={metrics['fpr']:.3f}"
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.3, fontweight="bold", pad=8)

    fig.suptitle(
        "Pairwise LightGBM Confusion Matrices Across Evaluation Regimes",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(data: dict[str, Any], out_path: Path) -> None:
    """Plot a ranked ConPlag v2 scoreboard comparing F1, Precision, Recall, AUC, and FPR."""
    cv = data["conplag_v2_cv"]
    ranked = sorted(
        MODEL_ORDER,
        key=lambda key: cv[key]["micro_oof"]["f1"],
        reverse=True,
    )
    labels = [MODEL_META[k]["short"].replace("\n", " ") for k in ranked]
    y = np.arange(len(ranked))

    f1_vals = [cv[k]["micro_oof"]["f1"] for k in ranked]
    prec_vals = [cv[k]["micro_oof"]["precision"] for k in ranked]
    rec_vals = [cv[k]["micro_oof"]["recall"] for k in ranked]
    auc_vals = [cv[k]["micro_oof"]["roc_auc"] for k in ranked]
    fpr_vals = [cv[k]["micro_oof"]["fpr"] for k in ranked]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.4),
        gridspec_kw={"width_ratios": [1.85, 1.0]},
    )

    bar_h = 0.19
    series = [
        ("ROC AUC", auc_vals, "#1b4965", -1.5 * bar_h),
        ("F1 Score", f1_vals, "#2a9d8f", -0.5 * bar_h),
        ("Precision", prec_vals, "#62b6cb", 0.5 * bar_h),
        ("Recall", rec_vals, "#e9c46a", 1.5 * bar_h),
    ]

    for metric_name, vals, color, dy in series:
        bars = ax_left.barh(
            y + dy,
            vals,
            height=bar_h,
            label=metric_name,
            color=color,
            edgecolor="#222222",
            linewidth=0.45,
        )
        for bar, val in zip(bars, vals):
            ax_left.text(
                val + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{val * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=7.3,
            )

    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax_left.invert_yaxis()
    ax_left.set_xlim(0.0, 1.08)
    ax_left.set_xlabel("Score (Higher is Better)")
    ax_left.set_title("ConPlag v2 5-Fold CV: Detection Quality", pad=22)
    ax_left.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8.3,
        ncol=4,
        frameon=False,
    )

    fpr_colors = ["#2a9d8f" if v <= 0.10 else "#e76f51" for v in fpr_vals]
    bars_fpr = ax_right.barh(
        y,
        fpr_vals,
        height=0.52,
        color=fpr_colors,
        edgecolor="#222222",
        linewidth=0.6,
    )
    for bar, val in zip(bars_fpr, fpr_vals):
        ax_right.text(
            val + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
        )

    ax_right.axvline(0.05, color="#555555", linestyle="--", linewidth=1.0, label="5% Low-FPR Target")
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])
    ax_right.invert_yaxis()
    ax_right.set_xlim(0.0, 0.34)
    ax_right.set_xlabel("False Positive Rate (Lower is Better)")
    ax_right.set_title("False Positive Rate (F1-Optimal Point)", pad=22)
    ax_right.legend(loc="upper right", fontsize=8.2)

    fig.suptitle(
        "ConPlag v2 Problem-Disjoint 5-Fold CV Model Scoreboard (Ranked by F1)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_cnn_vs_lgbm_held_out(data: dict[str, Any], out_path: Path) -> None:
    """Plot held out Fold 4 ConPlag v2 and Zero Shot IR Plag comparison for LightGBM vs CNN."""
    held_out = data.get("held_out_cnn_comparison")
    if not isinstance(held_out, dict) or not isinstance(held_out.get("models"), dict):
        return

    models = held_out["models"]
    lgbm = models["pairwise_lgbm_3fold"]
    cnn = models["pairwise_cnn"]

    metrics_keys = ("f1", "precision", "recall", "roc_auc", "fpr")
    metric_labels = ("F1", "Precision", "Recall", "ROC AUC", "FPR")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    x = np.arange(len(metrics_keys))
    width = 0.36

    panels = (
        (axes[0], "conplag_v2_fold4", f"ConPlag v2 Unseen Fold 4 (n={held_out['n_test']})"),
        (axes[1], "ir_plag", f"IR Plag Zero Shot (n={held_out['n_ir_plag']})"),
    )

    for ax, split_key, title in panels:
        lgbm_vals = [float(lgbm[split_key][k]) for k in metrics_keys]
        cnn_vals = [float(cnn[split_key][k]) for k in metrics_keys]

        bars_lgbm = ax.bar(
            x - width / 2,
            lgbm_vals,
            width,
            label="Pairwise LightGBM (3 Fold)",
            color="#1b4965",
            edgecolor="#222222",
            linewidth=0.6,
        )
        bars_cnn = ax.bar(
            x + width / 2,
            cnn_vals,
            width,
            label="Pairwise CNN (Exported)",
            color="#d97706",
            edgecolor="#222222",
            linewidth=0.6,
        )
        for bars in (bars_lgbm, bars_cnn):
            for bar in bars:
                val = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.015,
                    f"{val * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0.0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle(
        "Held Out Same Split Comparison: Pairwise LightGBM vs Pairwise CNN (Trained on Folds 0, 1, 2)",
        fontsize=12.0,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Load metrics_summary.json and render all evaluation figures."""
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    with METRICS_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    _configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        (FIGURES_DIR / "model_comparison.png", plot_model_comparison),
        (FIGURES_DIR / "roc_curves.png", plot_roc_curves),
        (FIGURES_DIR / "pr_curves.png", plot_pr_curves),
        (FIGURES_DIR / "ir_plag_levels.png", plot_ir_plag_levels),
        (FIGURES_DIR / "boilerplate_ablation.png", plot_boilerplate_ablation),
        (FIGURES_DIR / "confusion_matrices.png", plot_confusion_matrices),
    ]
    if isinstance(data.get("held_out_cnn_comparison"), dict):
        outputs.append((FIGURES_DIR / "cnn_vs_lgbm_held_out.png", plot_cnn_vs_lgbm_held_out))

    for out_path, plot_fn in outputs:
        plot_fn(data, out_path)
        size_kb = out_path.stat().st_size / 1024.0
        print(f"Generated {out_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
