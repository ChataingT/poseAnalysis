"""
Statistical analysis: outlier detection, group comparisons, and correlations.

Outlier detection
-----------------
Two complementary methods are applied per metric:
  - Z-score: |z| > z_threshold  (default 3.0)
  - IQR: value < Q1 - iqr_mult*IQR  or  > Q3 + iqr_mult*IQR  (default 3.0)
A value is flagged if either method detects it as an outlier.
Flagged values are replaced with NaN before statistical tests.

Statistical tests
-----------------
Binary features (diagnosis, gender) → Mann-Whitney U + Cohen's d
Continuous features (age, ADOS_2_TOTAL)  → Spearman correlation

Multiple comparisons → Benjamini-Hochberg FDR correction, applied separately
for each (feature, variant) combination.

Output structure
----------------
  stats/binary_{feature}_{variant}.csv
  stats/continuous_{feature}_{variant}.csv
  outliers/outlier_report.csv
  outliers/subject_summary.csv
  outliers/metric_summary.csv
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from .load import get_metric_columns, parse_metric_column

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────

BINARY_FEATURES: dict[str, dict] = {
    "diagnosis": {
        "col": "diagnosis",
        "group1": "ASD",
        "group2": "TD",
        "label1": "ASD",
        "label2": "TD",
    },
    "gender": {
        "col": "gender",
        "group1": "Male",
        "group2": "Female",
        "label1": "Male",
        "label2": "Female",
    },
}

CONTINUOUS_FEATURES: dict[str, dict] = {
    "age": {"col": "Ados_2_Age", "label": "Age (years)"},
    "ados_total": {"col": "ADOS_2_TOTAL", "label": "ADOS-2 Total Score"},
}


# ─────────────────────────────────────────────────────────────
# Outlier detection
# ─────────────────────────────────────────────────────────────

def detect_outliers(
    df: pd.DataFrame,
    metric_cols: list[str],
    z_threshold: float = 3.0,
    iqr_mult: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect per-metric outliers using z-score and IQR methods.

    Args:
        df: Merged dataset with subject rows.
        metric_cols: Column names of metrics to check.
        z_threshold: Z-score threshold for outlier flag.
        iqr_mult: IQR multiplier for outlier flag.

    Returns:
        outlier_mask: Boolean DataFrame (True = outlier), shape (n_subjects, n_metrics).
        outlier_report: Long-form DataFrame of all flagged entries.
        subject_summary: Per-subject count of flagged metrics.
        metric_summary: Per-metric count and percentage of flagged subjects.
    """
    outlier_mask = pd.DataFrame(False, index=df.index, columns=metric_cols)
    records: list[dict] = []

    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) < 5:
            continue

        # Z-score
        z_arr = np.abs(scipy_stats.zscore(vals.values, nan_policy="omit"))
        z = pd.Series(z_arr, index=vals.index)
        z_flags = z > z_threshold

        # IQR
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_flags = (vals < q1 - iqr_mult * iqr) | (vals > q3 + iqr_mult * iqr)
        else:
            iqr_flags = pd.Series(False, index=vals.index)

        combined = z_flags | iqr_flags
        outlier_mask.loc[vals.index, col] = combined

        parsed = parse_metric_column(col)
        for idx in combined[combined].index:
            records.append(
                {
                    "subject_uuid": df.loc[idx, "uuid"]
                    if "uuid" in df.columns
                    else str(idx),
                    "diagnosis": df.loc[idx, "diagnosis"]
                    if "diagnosis" in df.columns
                    else None,
                    "metric_col": col,
                    "base_metric": parsed["base"],
                    "variant": parsed["variant"],
                    "stat_type": parsed["stat_type"],
                    "value": float(vals[idx]),
                    "z_score": float(z[idx]),
                    "z_outlier": bool(z_flags[idx]),
                    "iqr_outlier": bool(iqr_flags[idx]),
                }
            )

    outlier_report = pd.DataFrame(records)

    # Per-subject summary
    n_flagged_per_subject = outlier_mask.sum(axis=1)
    subject_summary = pd.DataFrame(
        {
            "subject_uuid": df["uuid"] if "uuid" in df.columns else df.index,
            "diagnosis": df["diagnosis"] if "diagnosis" in df.columns else None,
            "n_outlier_metrics": n_flagged_per_subject.values,
        }
    )

    # Per-metric summary
    n_outliers_per_metric = outlier_mask.sum(axis=0)
    metric_summary = pd.DataFrame(
        {
            "metric_col": metric_cols,
            "n_outliers": n_outliers_per_metric.values,
            "pct_outliers": (n_outliers_per_metric.values / len(df)) * 100,
        }
    )

    n_flagged = outlier_report.shape[0]
    logger.info(
        f"Outlier detection: {n_flagged} flagged entries across "
        f"{len(metric_cols)} metrics and {len(df)} subjects"
    )
    return outlier_mask, outlier_report, subject_summary, metric_summary


def apply_outlier_mask(df: pd.DataFrame, outlier_mask: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with outlier values replaced by NaN."""
    df_clean = df.copy()
    for col in outlier_mask.columns:
        if col in df_clean.columns:
            df_clean.loc[outlier_mask[col], col] = np.nan
    return df_clean


# ─────────────────────────────────────────────────────────────
# Individual statistical tests
# ─────────────────────────────────────────────────────────────

def _cohen_d(g1: pd.Series, g2: pd.Series) -> float:
    """Compute Cohen's d (pooled standard deviation)."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_var = (
        (n1 - 1) * g1.var() + (n2 - 1) * g2.var()
    ) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return np.nan
    return (g1.mean() - g2.mean()) / pooled_std


def _binary_test(
    df: pd.DataFrame,
    metric_col: str,
    feature_cfg: dict,
    min_n: int = 5,
) -> dict | None:
    """Mann-Whitney U test + Cohen's d between two groups.

    Returns None if either group has fewer than min_n observations.
    """
    col_name = feature_cfg["col"]
    g1_label, g2_label = feature_cfg["group1"], feature_cfg["group2"]

    vals = df[metric_col]
    g1 = vals[df[col_name] == g1_label].dropna()
    g2 = vals[df[col_name] == g2_label].dropna()

    if len(g1) < min_n or len(g2) < min_n:
        return None

    u_stat, p_val = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    cohens_d = _cohen_d(g1, g2)

    return {
        f"n_{g1_label}": len(g1),
        f"mean_{g1_label}": g1.mean(),
        f"std_{g1_label}": g1.std(),
        f"median_{g1_label}": g1.median(),
        f"n_{g2_label}": len(g2),
        f"mean_{g2_label}": g2.mean(),
        f"std_{g2_label}": g2.std(),
        f"median_{g2_label}": g2.median(),
        "n_total_used": len(g1) + len(g2),
        "mwu_statistic": u_stat,
        "pvalue": p_val,
        "cohens_d": cohens_d,
    }


def _continuous_test(
    df: pd.DataFrame,
    metric_col: str,
    feature_cfg: dict,
    min_n: int = 5,
) -> dict | None:
    """Spearman correlation between a metric and a continuous feature.

    Returns None if fewer than min_n paired valid observations.
    """
    feat_col = feature_cfg["col"]
    paired = df[[feat_col, metric_col]].dropna()
    if len(paired) < min_n:
        return None

    rho, p_val = scipy_stats.spearmanr(paired[feat_col], paired[metric_col])

    return {
        "n_total_used": len(paired),
        "spearman_rho": float(rho),
        "pvalue": float(p_val),
    }


# ─────────────────────────────────────────────────────────────
# Full analysis
# ─────────────────────────────────────────────────────────────

def _run_for_variant(
    df: pd.DataFrame,
    variant: Literal["raw", "norm"],
    feature_name: str,
    feature_cfg: dict,
    test_type: Literal["binary", "continuous"],
    alpha: float,
    min_n: int,
) -> pd.DataFrame:
    """Run one feature × variant block. Returns a stats DataFrame."""
    metric_cols = get_metric_columns(df, variant=variant)
    rows: list[dict] = []

    for col in metric_cols:
        parsed = parse_metric_column(col)
        base = parsed["base"]
        stat_type = parsed["stat_type"]

        if test_type == "binary":
            result = _binary_test(df, col, feature_cfg, min_n=min_n)
        else:
            result = _continuous_test(df, col, feature_cfg, min_n=min_n)

        if result is None:
            continue

        row = {
            "metric_col": col,
            "base_metric": base,
            "variant": variant,
            "stat_type": stat_type,
        }
        row.update(result)
        rows.append(row)

    if not rows:
        logger.warning(f"  No valid tests for {feature_name} / {variant}")
        return pd.DataFrame()

    res = pd.DataFrame(rows)

    # FDR correction
    reject, pvals_corr, _, _ = multipletests(
        res["pvalue"].values, method="fdr_bh", alpha=alpha
    )
    res["pvalue_fdr"] = pvals_corr
    res["significant_fdr"] = reject

    return res.sort_values("pvalue").reset_index(drop=True)


def run_all_statistics(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_n: int = 5,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run all statistical tests across features and metric variants.

    Args:
        df: Merged dataset (with outliers already replaced by NaN if desired).
        alpha: Significance threshold for FDR correction.
        min_n: Minimum number of valid observations per group/test.

    Returns:
        Nested dict: results[feature_name][variant] → pd.DataFrame
        feature_name ∈ {"diagnosis", "gender", "age", "ados_total"}
        variant      ∈ {"raw", "norm"}
    """
    results: dict[str, dict[str, pd.DataFrame]] = {}

    all_features = [
        ("binary",     "diagnosis",  BINARY_FEATURES["diagnosis"]),
        ("binary",     "gender",     BINARY_FEATURES["gender"]),
        ("continuous", "age",        CONTINUOUS_FEATURES["age"]),
        ("continuous", "ados_total", CONTINUOUS_FEATURES["ados_total"]),
    ]

    for test_type, feat_name, feat_cfg in all_features:
        results[feat_name] = {}
        for variant in ("raw", "norm"):
            logger.info(f"  Running {test_type} test: {feat_name} × {variant}")
            res = _run_for_variant(
                df, variant, feat_name, feat_cfg, test_type, alpha, min_n
            )
            results[feat_name][variant] = res

    return results


def build_top_metrics_table(
    results: dict[str, dict[str, pd.DataFrame]],
    n_top: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Combine results across all features and return the top significant metrics.

    Args:
        results: Output of run_all_statistics.
        n_top: Number of top metrics per (feature, variant) to include.
        alpha: FDR significance threshold.

    Returns:
        DataFrame with columns: feature, variant, metric_col, base_metric,
        stat_type, effect_size, pvalue_fdr, significant_fdr.
    """
    rows: list[dict] = []
    for feat_name, variant_dict in results.items():
        for variant, df_stats in variant_dict.items():
            if df_stats.empty:
                continue
            sig = df_stats[df_stats["pvalue_fdr"] < alpha].copy()

            # Select effect-size column
            if "cohens_d" in sig.columns:
                sig["effect_size"] = sig["cohens_d"]
            elif "spearman_rho" in sig.columns:
                sig["effect_size"] = sig["spearman_rho"]
            else:
                sig["effect_size"] = np.nan

            sig["abs_effect_size"] = sig["effect_size"].abs()
            top = sig.nlargest(n_top, "abs_effect_size")

            for _, row in top.iterrows():
                rows.append(
                    {
                        "feature": feat_name,
                        "variant": variant,
                        "metric_col": row["metric_col"],
                        "base_metric": row["base_metric"],
                        "stat_type": row["stat_type"],
                        "effect_size": row["effect_size"],
                        "abs_effect_size": row["abs_effect_size"],
                        "pvalue_fdr": row["pvalue_fdr"],
                        "significant_fdr": row["significant_fdr"],
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["feature", "variant", "abs_effect_size"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
