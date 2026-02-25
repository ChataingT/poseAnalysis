"""
Visualization module for pose analysis.

Generates the following figures:

Outlier diagnostics
    outliers/outlier_boxplot_{raw|norm}.png    — metric distributions with flagged subjects highlighted
    outliers/per_subject_outlier_count.png      — bar chart of outlier count per subject

Overview summaries (per variant)
    heatmap_{raw|norm}.png                      — effect-size heatmap (all metrics × all features)
    top20_{feature}_{raw|norm}.png              — top-20 metrics by |effect size| per feature
    volcano_{feature}_{raw|norm}.png            — volcano plot for each feature

Social metrics panels
    social_panel_{raw|norm}.png                 — violin / scatter for social-interaction metrics

Per-significant-metric individual plots (FDR < alpha)
    violin/{feature}/{base_metric}_{variant}.png
    scatter/{feature}/{base_metric}_{variant}.png
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster use
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from .load import get_metric_columns, parse_metric_column
from .stats import BINARY_FEATURES, CONTINUOUS_FEATURES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DPI = 150
PALETTE_DIAGNOSIS = {"ASD": "#E74C3C", "TD": "#2E86AB"}
PALETTE_GENDER = {"Male": "#3498DB", "Female": "#E91E8C"}
PALETTE_BINARY = {**PALETTE_DIAGNOSIS, **PALETTE_GENDER}

SOCIAL_METRICS = [
    "facingness",
    "interpersonal_distance_centroid",
    "interpersonal_distance_trunk",
    "interpersonal_approach",
    "congruent_motion",
    "agitation_global_ke",
]

CHILD_MOTION_METRICS = [
    "child_speed_centroid",
    "child_speed_trunk",
    "child_acceleration_centroid",
    "child_acceleration_trunk",
    "child_kinetic_energy",
]

CHILD_KP_METRICS_ORDER = [
    "child_speed_kp_nose",
    "child_speed_kp_left_eye", "child_speed_kp_right_eye",
    "child_speed_kp_left_ear", "child_speed_kp_right_ear",
    "child_speed_kp_left_shoulder", "child_speed_kp_right_shoulder",
    "child_speed_kp_left_elbow", "child_speed_kp_right_elbow",
    "child_speed_kp_left_wrist", "child_speed_kp_right_wrist",
    "child_speed_kp_left_hip", "child_speed_kp_right_hip",
    "child_speed_kp_left_knee", "child_speed_kp_right_knee",
    "child_speed_kp_left_ankle", "child_speed_kp_right_ankle",
]

CLINICIAN_MOTION_METRICS = [
    "clinician_speed_centroid",
    "clinician_speed_trunk",
    "clinician_acceleration_centroid",
    "clinician_acceleration_trunk",
    "clinician_kinetic_energy",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def _clean_label(col: str) -> str:
    """Return human-readable label from metric column name."""
    parsed = parse_metric_column(col)
    return parsed["base"].replace("_", " ")


# ─────────────────────────────────────────────────────────────
# Outlier diagnostic plots
# ─────────────────────────────────────────────────────────────

def _boxplot_group(
    df: pd.DataFrame,
    metric_cols: list[str],
    outlier_mask: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Box-plot of a group of metrics with outliers highlighted."""
    n = len(metric_cols)
    if n == 0:
        return

    cols_present = [c for c in metric_cols if c in df.columns]
    if not cols_present:
        return

    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), 5))

    data = df[cols_present].copy()
    labels = [_clean_label(c) for c in cols_present]

    ax.boxplot(
        [data[c].dropna().values for c in cols_present],
        labels=labels,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1", alpha=0.7),
        medianprops=dict(color="#154360", linewidth=2),
    )

    # Overlay outlier points
    for xi, col in enumerate(cols_present, start=1):
        if col not in outlier_mask.columns:
            continue
        outlier_vals = df.loc[outlier_mask[col], col].dropna()
        if len(outlier_vals) > 0:
            ax.scatter(
                [xi] * len(outlier_vals),
                outlier_vals.values,
                color="#E74C3C",
                s=40,
                zorder=5,
                alpha=0.8,
                label="outlier" if xi == 1 else None,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
                   markersize=8, label="Outlier")
    ]
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_outlier_report(
    df: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    subject_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    output_dir: Path,
    variant: Literal["raw", "norm"] = "raw",
) -> None:
    """Generate outlier diagnostic plots for a given variant."""
    out = _ensure_dir(output_dir / "outliers")
    metric_cols = get_metric_columns(df, variant=variant, stat_type="mean_of_mean")

    # Group boxplots by metric category
    groups = {
        "Social metrics": [c for c in metric_cols if any(sm in c for sm in SOCIAL_METRICS)],
        "Child motion": [c for c in metric_cols if any(cm in c for cm in CHILD_MOTION_METRICS)],
        "Child keypoints": [c for c in metric_cols if "child_speed_kp" in c],
        "Clinician": [c for c in metric_cols if c.startswith("clinician")],
    }

    for group_name, gcols in groups.items():
        if not gcols:
            continue
        fname = f"outlier_boxplot_{variant}_{group_name.lower().replace(' ', '_')}.png"
        _boxplot_group(df, gcols, outlier_mask, f"Outliers — {group_name} ({variant})", out / fname)
        logger.info(f"  Saved {fname}")

    # Per-subject outlier count bar chart
    if not subject_summary.empty and "n_outlier_metrics" in subject_summary.columns:
        fig, ax = plt.subplots(figsize=(max(10, len(subject_summary) * 0.15), 4))
        ss = subject_summary.sort_values("n_outlier_metrics", ascending=False).reset_index(drop=True)
        colors = [PALETTE_DIAGNOSIS.get(d, "#999") for d in ss.get("diagnosis", [""] * len(ss))]
        ax.bar(range(len(ss)), ss["n_outlier_metrics"], color=colors, alpha=0.8)
        ax.set_xlabel("Subject (ranked by n outliers)")
        ax.set_ylabel("N metrics flagged as outlier")
        ax.set_title(f"Per-subject outlier count ({variant})", fontweight="bold")
        # legend for diagnosis colors
        handles = [
            mpatches.Patch(facecolor=PALETTE_DIAGNOSIS["ASD"], label="ASD"),
            mpatches.Patch(facecolor=PALETTE_DIAGNOSIS["TD"], label="TD"),
        ]
        ax.legend(handles=handles)
        plt.tight_layout()
        fig.savefig(out / f"per_subject_outlier_count_{variant}.png", dpi=DPI)
        plt.close(fig)
        logger.info(f"  Saved per_subject_outlier_count_{variant}.png")


# ─────────────────────────────────────────────────────────────
# Overview: heatmap of effect sizes
# ─────────────────────────────────────────────────────────────

def plot_heatmap(
    results: dict[str, dict[str, pd.DataFrame]],
    variant: Literal["raw", "norm"],
    output_path: Path,
    stat_type: str = "mean_of_mean",
    n_top_metrics: int = 50,
) -> None:
    """Heatmap of signed effect sizes (features × metrics), significant cells marked."""
    feature_order = ["diagnosis", "gender", "age", "ados_total"]
    feature_labels = {
        "diagnosis": "Diagnosis\n(ASD>TD)",
        "gender": "Gender\n(Male>Female)",
        "age": "Age\n(Spearman ρ)",
        "ados_total": "ADOS Total\n(Spearman ρ)",
    }

    # Collect effect sizes and FDR p-values
    eff_dict: dict[str, dict[str, float]] = {}
    fdr_dict: dict[str, dict[str, float]] = {}

    for feat in feature_order:
        if feat not in results or variant not in results[feat]:
            continue
        df_s = results[feat][variant]
        if df_s.empty:
            continue
        sub = df_s[df_s["stat_type"] == stat_type]

        eff_col = "cohens_d" if "cohens_d" in sub.columns else (
            "spearman_rho" if "spearman_rho" in sub.columns else None
        )
        if eff_col is None:
            continue

        for _, row in sub.iterrows():
            m = row["base_metric"]
            eff_dict.setdefault(m, {})[feat] = row[eff_col]
            fdr_dict.setdefault(m, {})[feat] = row["pvalue_fdr"]

    if not eff_dict:
        logger.warning(f"  Heatmap: no data for variant={variant}")
        return

    eff_df = pd.DataFrame(eff_dict).T.reindex(columns=feature_order)
    fdr_df = pd.DataFrame(fdr_dict).T.reindex(columns=feature_order)

    # Select top metrics by mean |effect size|
    mean_abs = eff_df.abs().mean(axis=1)
    top_metrics = mean_abs.nlargest(n_top_metrics).index.tolist()
    eff_df = eff_df.loc[top_metrics]
    fdr_df = fdr_df.loc[top_metrics]

    # Cluster rows
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    try:
        data_for_cluster = eff_df.fillna(0).values
        dist = pdist(data_for_cluster, metric="euclidean")
        link = linkage(dist, method="ward")
        order = leaves_list(link)
        eff_df = eff_df.iloc[order]
        fdr_df = fdr_df.iloc[order]
    except Exception:
        pass  # Keep original order if clustering fails

    fig, ax = plt.subplots(figsize=(6, max(8, len(eff_df) * 0.28)))
    vmax = min(eff_df.abs().max().max(), 1.5)
    sns.heatmap(
        eff_df,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Effect size (Cohen's d / Spearman ρ)", "shrink": 0.5},
        yticklabels=[m.replace("_", " ") for m in eff_df.index],
        xticklabels=[feature_labels.get(f, f) for f in feature_order],
        annot=False,
    )
    # Overlay significance stars
    for yi, metric in enumerate(eff_df.index):
        for xi, feat in enumerate(feature_order):
            if feat not in fdr_df.columns:
                continue
            p = fdr_df.loc[metric, feat]
            if not np.isnan(p) and p < 0.05:
                ax.text(
                    xi + 0.5, yi + 0.5, _stars(p),
                    ha="center", va="center", fontsize=7, color="black",
                )

    ax.set_title(
        f"Effect-size heatmap ({variant}, {stat_type})\nTop {n_top_metrics} metrics",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────
# Overview: top-20 barplot
# ─────────────────────────────────────────────────────────────

def plot_top_metrics(
    df_stats: pd.DataFrame,
    feature_name: str,
    variant: str,
    stat_type: str,
    output_path: Path,
    n_top: int = 20,
    alpha: float = 0.05,
) -> None:
    """Horizontal bar chart of top-N metrics by |effect size|."""
    sub = df_stats[df_stats["stat_type"] == stat_type].copy()
    if sub.empty:
        return

    eff_col = "cohens_d" if "cohens_d" in sub.columns else (
        "spearman_rho" if "spearman_rho" in sub.columns else None
    )
    if eff_col is None:
        return

    sub["abs_eff"] = sub[eff_col].abs()
    top = sub.nlargest(n_top, "abs_eff").sort_values("abs_eff")

    colors = [
        ("#E74C3C" if v > 0 else "#2980B9") if fdr < alpha else "#BDC3C7"
        for v, fdr in zip(top[eff_col], top["pvalue_fdr"])
    ]

    fig, ax = plt.subplots(figsize=(7, max(5, len(top) * 0.35)))
    bars = ax.barh(
        [m.replace("_", " ") for m in top["base_metric"]],
        top[eff_col],
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "Cohen's d (ASD vs TD)" if eff_col == "cohens_d" else "Spearman ρ"
    )
    ax.set_title(
        f"Top {n_top} metrics — {feature_name} ({variant}, {stat_type})",
        fontweight="bold",
    )
    # Annotate with significance
    for bar, fdr in zip(bars, top["pvalue_fdr"]):
        if fdr < alpha:
            ax.text(
                bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                _stars(fdr),
                va="center",
                ha="left" if bar.get_width() >= 0 else "right",
                fontsize=9,
            )
    handles = [
        mpatches.Patch(facecolor="#E74C3C", label="Positive effect (FDR<0.05)"),
        mpatches.Patch(facecolor="#2980B9", label="Negative effect (FDR<0.05)"),
        mpatches.Patch(facecolor="#BDC3C7", label="Not significant"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────
# Overview: volcano plot
# ─────────────────────────────────────────────────────────────

def plot_volcano(
    df_stats: pd.DataFrame,
    feature_name: str,
    variant: str,
    stat_type: str,
    output_path: Path,
    alpha: float = 0.05,
    n_label: int = 10,
) -> None:
    """Volcano plot: effect size vs -log10(FDR p-value)."""
    sub = df_stats[df_stats["stat_type"] == stat_type].copy()
    if sub.empty:
        return

    eff_col = "cohens_d" if "cohens_d" in sub.columns else (
        "spearman_rho" if "spearman_rho" in sub.columns else None
    )
    if eff_col is None:
        return

    sub["neg_log10_fdr"] = -np.log10(sub["pvalue_fdr"].clip(lower=1e-10))
    sub["sig"] = sub["pvalue_fdr"] < alpha

    fig, ax = plt.subplots(figsize=(8, 6))
    # Non-significant points
    ns = sub[~sub["sig"]]
    ax.scatter(ns[eff_col], ns["neg_log10_fdr"], c="#BDC3C7", s=20, alpha=0.6, label="ns")
    # Positive effect
    pos = sub[sub["sig"] & (sub[eff_col] > 0)]
    ax.scatter(pos[eff_col], pos["neg_log10_fdr"], c="#E74C3C", s=40, alpha=0.85, label="sig+")
    # Negative effect
    neg = sub[sub["sig"] & (sub[eff_col] < 0)]
    ax.scatter(neg[eff_col], neg["neg_log10_fdr"], c="#2980B9", s=40, alpha=0.85, label="sig−")

    # Label top points by -log10 FDR
    top_pts = sub.nlargest(n_label, "neg_log10_fdr")
    for _, row in top_pts.iterrows():
        ax.annotate(
            row["base_metric"].replace("_", " "),
            xy=(row[eff_col], row["neg_log10_fdr"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=6.5,
            alpha=0.9,
        )

    # Threshold line
    ax.axhline(-np.log10(alpha), color="gray", linestyle="--", linewidth=1, label=f"FDR={alpha}")
    ax.set_xlabel(
        "Cohen's d (group1 > group2)" if eff_col == "cohens_d" else "Spearman ρ",
        fontsize=10,
    )
    ax.set_ylabel("−log₁₀(FDR p-value)", fontsize=10)
    ax.set_title(
        f"Volcano — {feature_name} ({variant}, {stat_type})",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────
# Social metrics panel
# ─────────────────────────────────────────────────────────────

def plot_social_panel(
    df: pd.DataFrame,
    results: dict[str, dict[str, pd.DataFrame]],
    variant: Literal["raw", "norm"],
    stat_type: str,
    output_path: Path,
    alpha: float = 0.05,
) -> None:
    """Grid of violin plots for social/dyadic metrics vs diagnosis."""
    metric_cols = [
        c for c in get_metric_columns(df, variant=variant, stat_type=stat_type)
        if parse_metric_column(c)["base"] in SOCIAL_METRICS
    ]
    if not metric_cols:
        return

    diag_stats = results.get("diagnosis", {}).get(variant, pd.DataFrame())
    n_cols = 3
    n_rows = int(np.ceil(len(metric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, metric_cols):
        base = parse_metric_column(col)["base"]
        sub = df[["diagnosis", col]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        # Get FDR p-value
        p_fdr = np.nan
        if not diag_stats.empty:
            row = diag_stats[diag_stats["metric_col"] == col]
            if not row.empty:
                p_fdr = row.iloc[0]["pvalue_fdr"]

        order = ["ASD", "TD"]
        palette = {g: PALETTE_DIAGNOSIS.get(g, "#999") for g in order}
        try:
            sns.violinplot(
                data=sub, x="diagnosis", y=col, order=order,
                palette=palette, ax=ax, inner="box", alpha=0.7,
            )
            sns.stripplot(
                data=sub, x="diagnosis", y=col, order=order,
                palette=palette, ax=ax, size=3, alpha=0.5, jitter=True,
            )
        except Exception:
            pass

        title = base.replace("_", " ")
        if not np.isnan(p_fdr):
            sig_marker = f" {_stars(p_fdr)}" if p_fdr < alpha else ""
            title += f"\nFDR={p_fdr:.3f}{sig_marker}"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=8)

    # Hide unused axes
    for ax in axes[len(metric_cols):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Social / Dyadic Metrics vs Diagnosis ({variant}, {stat_type})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────
# Individual metric violin / scatter
# ─────────────────────────────────────────────────────────────

def plot_violin_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    feature_name: str,
    feature_cfg: dict,
    pvalue_fdr: float,
    output_path: Path,
    alpha: float = 0.05,
) -> None:
    """Violin + strip for a binary comparison on one metric."""
    col_name = feature_cfg["col"]
    g1, g2 = feature_cfg["group1"], feature_cfg["group2"]
    sub = df[[col_name, metric_col]].dropna()
    sub = sub[sub[col_name].isin([g1, g2])]
    if sub.empty:
        return

    palette = {k: PALETTE_BINARY.get(k, "#999") for k in [g1, g2]}
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.violinplot(
        data=sub, x=col_name, y=metric_col, order=[g1, g2],
        palette=palette, ax=ax, inner="box", alpha=0.75,
    )
    sns.stripplot(
        data=sub, x=col_name, y=metric_col, order=[g1, g2],
        palette=palette, ax=ax, size=3.5, alpha=0.55, jitter=True,
    )

    base = parse_metric_column(metric_col)["base"]
    sig_txt = _stars(pvalue_fdr) if pvalue_fdr < alpha else f"ns (FDR={pvalue_fdr:.3f})"
    ax.set_title(
        f"{base.replace('_', ' ')}\n{feature_name}: {sig_txt}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Value")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    feature_name: str,
    feature_cfg: dict,
    spearman_rho: float,
    pvalue_fdr: float,
    output_path: Path,
    alpha: float = 0.05,
) -> None:
    """Scatter plot for a continuous feature vs one metric."""
    feat_col = feature_cfg["col"]
    sub = df[[feat_col, metric_col, "diagnosis"]].dropna(subset=[feat_col, metric_col])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    # Color by diagnosis if available
    for diag, grp in sub.groupby("diagnosis"):
        ax.scatter(
            grp[feat_col], grp[metric_col],
            label=diag, color=PALETTE_DIAGNOSIS.get(diag, "#999"),
            s=30, alpha=0.7,
        )

    # Regression line
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from scipy.stats import linregress
            slope, intercept, *_ = linregress(sub[feat_col], sub[metric_col])
            xr = np.linspace(sub[feat_col].min(), sub[feat_col].max(), 100)
            ax.plot(xr, slope * xr + intercept, color="#555", linewidth=1.5, linestyle="--")
        except Exception:
            pass

    base = parse_metric_column(metric_col)["base"]
    sig_txt = f"ρ={spearman_rho:.3f}, FDR={pvalue_fdr:.3f} {_stars(pvalue_fdr)}"
    ax.set_title(
        f"{base.replace('_', ' ')}\nvs {feature_cfg.get('label', feature_name)}\n{sig_txt}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel(feature_cfg.get("label", feature_name))
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def generate_all_figures(
    df: pd.DataFrame,
    df_orig: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    subject_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    results: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    alpha: float = 0.05,
    max_individual_plots: int = 100,
) -> None:
    """Generate the complete figure suite.

    Args:
        df: Merged dataset with outliers replaced by NaN (used for violin/scatter/social plots).
        df_orig: Original merged dataset before outlier masking (used for outlier boxplots,
                 so that flagged values are still visible as red dots).
        outlier_mask: Boolean mask from detect_outliers.
        subject_summary: Per-subject outlier counts.
        metric_summary: Per-metric outlier counts.
        results: Output of run_all_statistics.
        output_dir: Root output directory.
        alpha: FDR significance threshold.
        max_individual_plots: Limit on individual metric plots to avoid thousands of files.
    """
    fig_dir = _ensure_dir(output_dir / "figures")

    for variant in ("raw", "norm"):
        logger.info(f"Generating figures for variant={variant} …")

        # ── Outlier diagnostics — use df_orig so flagged values are still present ──
        plot_outlier_report(df_orig, outlier_mask, subject_summary, metric_summary,
                            fig_dir, variant=variant)

        for stat_type in ("mean_of_mean", "pooled_std"):
            # ── Effect-size heatmap ──────────────────────────
            heatmap_path = fig_dir / f"heatmap_{variant}_{stat_type}.png"
            plot_heatmap(results, variant, heatmap_path, stat_type=stat_type)

            for feat_name, feat_dict in {**{k: v for k, v in BINARY_FEATURES.items()},
                                          **{k: v for k, v in CONTINUOUS_FEATURES.items()}}.items():
                if feat_name not in results or variant not in results[feat_name]:
                    continue
                df_s = results[feat_name][variant]
                if df_s.empty:
                    continue

                # ── Top-20 barplot ───────────────────────────
                top_path = fig_dir / f"top20_{feat_name}_{variant}_{stat_type}.png"
                plot_top_metrics(df_s, feat_name, variant, stat_type, top_path, alpha=alpha)

                # ── Volcano ──────────────────────────────────
                vol_path = fig_dir / f"volcano_{feat_name}_{variant}_{stat_type}.png"
                plot_volcano(df_s, feat_name, variant, stat_type, vol_path, alpha=alpha)

            # ── Social metrics panel ─────────────────────────
            social_path = fig_dir / f"social_panel_{variant}_{stat_type}.png"
            plot_social_panel(df, results, variant, stat_type, social_path, alpha=alpha)

        # ── Individual metric plots (only for mean_of_mean, FDR < alpha) ──
        n_plots = 0
        for feat_name in results:
            if variant not in results[feat_name]:
                continue
            df_s = results[feat_name][variant]
            if df_s.empty:
                continue

            sig_rows = df_s[
                (df_s["pvalue_fdr"] < alpha) & (df_s["stat_type"] == "mean_of_mean")
            ].copy()
            is_binary = feat_name in BINARY_FEATURES

            for _, row in sig_rows.iterrows():
                if n_plots >= max_individual_plots:
                    logger.info(
                        f"  Reached max_individual_plots={max_individual_plots}, stopping."
                    )
                    break

                col = row["metric_col"]
                base = row["base_metric"]
                pval_fdr = row["pvalue_fdr"]

                if is_binary:
                    feat_cfg = BINARY_FEATURES[feat_name]
                    out_path = (
                        fig_dir / "violin" / feat_name / f"{base}_{variant}.png"
                    )
                    plot_violin_for_metric(df, col, feat_name, feat_cfg, pval_fdr, out_path, alpha=alpha)
                else:
                    feat_cfg = CONTINUOUS_FEATURES[feat_name]
                    rho = row.get("spearman_rho", np.nan)
                    out_path = (
                        fig_dir / "scatter" / feat_name / f"{base}_{variant}.png"
                    )
                    plot_scatter_for_metric(df, col, feat_name, feat_cfg, rho, pval_fdr, out_path, alpha=alpha)

                n_plots += 1

    logger.info(f"Figure generation complete. {n_plots} individual metric plots saved.")
