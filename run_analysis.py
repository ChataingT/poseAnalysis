#!/usr/bin/env python3
"""
Exploratory correlation analysis: pose metrics vs child clinical features.

Usage
-----
    python run_analysis.py \\
        --csv /path/to/child_for_humanlisbet_paper_with_paths.csv \\
        --pose-records /path/to/pose_records/ \\
        --output-dir /path/to/results/

The script will create the following structure under --output-dir:
    merged_dataset.csv             Raw subject × metric table
    outliers/
        outlier_report.csv         All flagged (subject, metric) pairs
        subject_summary.csv        Per-subject outlier count
        metric_summary.csv         Per-metric outlier count
    stats/
        binary_diagnosis_raw.csv
        binary_diagnosis_norm.csv
        binary_gender_raw.csv
        binary_gender_norm.csv
        continuous_age_raw.csv
        continuous_age_norm.csv
        continuous_ados_total_raw.csv
        continuous_ados_total_norm.csv
        top_significant_metrics.csv
    figures/
        (see viz.py for full listing)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from within the pose_analysis directory
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pose_analysis"

from .load import load_dataset, get_metric_columns
from .stats import (
    detect_outliers,
    apply_outlier_mask,
    run_all_statistics,
    build_top_metrics_table,
    compute_total_distance,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
)
from .viz import generate_all_figures


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to child_for_humanlisbet_paper_with_paths.csv",
    )
    parser.add_argument(
        "--pose-records",
        required=True,
        type=Path,
        help="Path to the pose_records/ directory",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results"),
        type=Path,
        help="Output directory (created if not existing). Default: ./results",
    )
    parser.add_argument(
        "--alpha",
        default=0.05,
        type=float,
        help="FDR significance threshold (default: 0.05)",
    )
    parser.add_argument(
        "--z-threshold",
        default=3.0,
        type=float,
        help="Z-score threshold for outlier detection (default: 3.0)",
    )
    parser.add_argument(
        "--iqr-mult",
        default=3.0,
        type=float,
        help="IQR multiplier for outlier detection (default: 3.0)",
    )
    parser.add_argument(
        "--min-subjects",
        default=5,
        type=int,
        help="Minimum subjects per group for a test to be run (default: 5)",
    )
    parser.add_argument(
        "--max-individual-plots",
        default=150,
        type=int,
        help="Maximum number of individual violin/scatter plots to generate (default: 150)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (useful for quick stats-only runs)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level),
        stream=sys.stdout,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("pose_analysis")

    # Validate inputs
    if not args.csv.exists():
        logger.error(f"CSV not found: {args.csv}")
        sys.exit(1)
    if not args.pose_records.exists():
        logger.error(f"pose_records directory not found: {args.pose_records}")
        sys.exit(1)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stats_dir = out / "stats"
    stats_dir.mkdir(exist_ok=True)
    outlier_dir = out / "outliers"
    outlier_dir.mkdir(exist_ok=True)

    # ── Step 1: Load dataset ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    df = load_dataset(args.csv, args.pose_records)
    if df.empty:
        logger.error("No subjects loaded. Check paths and CSV contents.")
        sys.exit(1)

    df.to_csv(out / "merged_dataset.csv", index=False)
    logger.info(f"Saved merged_dataset.csv  ({df.shape[0]} subjects × {df.shape[1]} columns)")

    all_metric_cols = get_metric_columns(df)
    logger.info(f"Metric columns: {len(all_metric_cols)}")

    # ── Step 2: Outlier detection ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Outlier detection")
    logger.info("=" * 60)

    outlier_mask, outlier_report, subject_summary, metric_summary = detect_outliers(
        df,
        all_metric_cols,
        z_threshold=args.z_threshold,
        iqr_mult=args.iqr_mult,
    )

    outlier_report.to_csv(outlier_dir / "outlier_report.csv", index=False)
    subject_summary.to_csv(outlier_dir / "subject_summary.csv", index=False)
    metric_summary.to_csv(outlier_dir / "metric_summary.csv", index=False)

    top_outlier_subjects = subject_summary.nlargest(10, "n_outlier_metrics")[
        ["subject_uuid", "diagnosis", "n_outlier_metrics"]
    ]
    logger.info(
        f"Top subjects by outlier count:\n"
        + top_outlier_subjects.to_string(index=False)
    )

    # Apply outlier mask (replace outlier values with NaN)
    df_clean = apply_outlier_mask(df, outlier_mask)
    logger.info("Outlier values replaced with NaN for statistical analysis.")

    # ── Step 3: Statistical analysis ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Statistical analysis")
    logger.info("=" * 60)

    results = run_all_statistics(
        df_clean,
        alpha=args.alpha,
        min_n=args.min_subjects,
    )

    # Save per-feature, per-variant CSVs
    for feat_name, variant_dict in results.items():
        for variant, df_stats in variant_dict.items():
            if df_stats.empty:
                continue
            is_binary = feat_name in BINARY_FEATURES
            prefix = "binary" if is_binary else "continuous"
            fname = f"{prefix}_{feat_name}_{variant}.csv"
            df_stats.to_csv(stats_dir / fname, index=False)
            n_sig = df_stats["significant_fdr"].sum()
            logger.info(
                f"  {fname}: {len(df_stats)} tests, {n_sig} significant (FDR < {args.alpha})"
            )

    # Top significant metrics summary
    top_df = build_top_metrics_table(results, n_top=20, alpha=args.alpha)
    top_df.to_csv(stats_dir / "top_significant_metrics.csv", index=False)

    # Print summary table
    logger.info("\n── Top significant metrics (sorted by feature × effect size) ──")
    if top_df.empty:
        logger.info("  No significant metrics found after FDR correction.")
    else:
        display_cols = ["feature", "variant", "base_metric", "stat_type",
                        "effect_size", "pvalue_fdr"]
        logger.info("\n" + top_df[display_cols].to_string(index=False))

    # ── Step 3a: Distance analysis ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3a: Distance analysis")
    logger.info("=" * 60)

    df_dist = compute_total_distance(df_clean)
    df_dist.to_csv(stats_dir / "distance_ranking.csv", index=False)
    logger.info(f"Saved distance_ranking.csv  ({len(df_dist)} subjects)")

    # Log top movers per metric
    log_metrics = [
        "dist_centroid_raw",        "dist_trunk_raw",
        "dist_centroid_norm_trunk", "dist_trunk_norm_trunk",
        "dist_centroid_norm_dur",   "dist_trunk_norm_dur",
    ]
    id_display = [c for c in ("code", "uuid", "diagnosis") if c in df_dist.columns]
    for metric in log_metrics:
        if metric not in df_dist.columns:
            continue
        top = df_dist.nlargest(5, metric)[id_display + [metric]]
        logger.info(f"\n  Top 5 by {metric}:\n{top.to_string(index=False)}")

    # ── Step 4: Figures ───────────────────────────────────────
    if not args.skip_figures:
        logger.info("=" * 60)
        logger.info("STEP 4: Generating figures")
        logger.info("=" * 60)

        generate_all_figures(
            df=df_clean,
            df_orig=df,
            outlier_mask=outlier_mask,
            subject_summary=subject_summary,
            metric_summary=metric_summary,
            results=results,
            output_dir=out,
            alpha=args.alpha,
            max_individual_plots=args.max_individual_plots,
        )
    else:
        logger.info("STEP 4: Skipped (--skip-figures)")

    # ── Final summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {out.resolve()}")

    # Print per-feature significance summary
    logger.info("\n── Significant metrics per feature (FDR < %.2f) ──" % args.alpha)
    for feat_name, variant_dict in results.items():
        for variant, df_stats in variant_dict.items():
            if df_stats.empty:
                continue
            sig_m = df_stats[df_stats["significant_fdr"] & (df_stats["stat_type"] == "mean_of_mean")]
            sig_s = df_stats[df_stats["significant_fdr"] & (df_stats["stat_type"] == "pooled_std")]
            logger.info(
                f"  {feat_name:15s} {variant:4s}  "
                f"{len(sig_m):3d} sig (mean_of_mean)  "
                f"{len(sig_s):3d} sig (pooled_std)"
            )


if __name__ == "__main__":
    main()
