"""
Load pose metrics from pose_records directory and merge with clinical metadata.

For each subject the pipeline aggregates segment-level metrics_summary.csv files
(which contain both raw and normalised statistics) into a single video-level
summary with one row per subject.

Column naming convention in the returned DataFrame:
    {base_metric}__{raw|norm}__{mean_of_mean|pooled_std}

Examples:
    child_speed_centroid__raw__mean_of_mean
    facingness__norm__pooled_std
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Clinical columns kept in the merged dataset
CLINICAL_COLS = [
    "uuid", "code", "src", "Ados_2_Age", "Ados_2_Module",
    "ADOS_2_TOTAL", "Visite", "diagnosis_raw", "gender",
    "sujet_id", "diagnosis",
]


def parse_metric_column(col: str) -> dict:
    """Parse a metric column name into its components.

    Example:
        'child_speed_centroid__raw__mean_of_mean'
        → {'base': 'child_speed_centroid', 'variant': 'raw', 'stat_type': 'mean_of_mean'}
    """
    parts = col.split("__")
    if len(parts) == 3:
        return {"base": parts[0], "variant": parts[1], "stat_type": parts[2]}
    return {"base": col, "variant": None, "stat_type": None}


def get_metric_columns(
    df: pd.DataFrame,
    variant: str | None = None,
    stat_type: str | None = None,
) -> list[str]:
    """Return metric column names, optionally filtered by variant and/or stat_type.

    Args:
        df: DataFrame with metric columns.
        variant: 'raw' or 'norm' (None = both).
        stat_type: 'mean_of_mean' or 'pooled_std' (None = both).

    Returns:
        List of matching column names.
    """
    cols = [c for c in df.columns if "__raw__" in c or "__norm__" in c]
    if variant is not None:
        cols = [c for c in cols if f"__{variant}__" in c]
    if stat_type is not None:
        cols = [c for c in cols if c.endswith(f"__{stat_type}")]
    return cols


def load_subject_metrics(pose_records_dir: Path, stem: str) -> pd.Series | None:
    """Load and aggregate segment-level metrics for one subject.

    Reads all segments/seg_*/metrics_summary.csv files and computes
    a weighted mean across segments (weight = number of valid frames).

    Args:
        pose_records_dir: Path to the pose_records directory.
        stem: Subdirectory name (equals results_path stem without .json).

    Returns:
        pd.Series with aggregated metrics, or None if directory/files missing.
    """
    subj_dir = pose_records_dir / stem
    if not subj_dir.exists():
        logger.warning(f"  Missing pose records dir: {subj_dir.name}")
        return None

    seg_files = sorted(subj_dir.glob("segments/seg_*/metrics_summary.csv"))
    if not seg_files:
        logger.warning(f"  No segment metric files in: {subj_dir.name}")
        return None

    dfs = []
    for f in seg_files:
        try:
            df = pd.read_csv(f, index_col=0)
            dfs.append(df)
        except Exception as exc:
            logger.warning(f"  Could not read {f}: {exc}")

    if not dfs:
        return None

    # Stack all segments; index = (seg_idx, metric_name)
    all_segs = pd.concat(dfs, keys=range(len(dfs)), names=["seg_idx", "metric"])

    metric_names = all_segs.index.get_level_values("metric").unique()
    records: dict[str, float] = {}

    for metric in metric_names:
        mdf = all_segs.xs(metric, level="metric")

        # --- Raw variant ---
        raw_w = mdf["raw_count"].fillna(0)
        raw_total = raw_w.sum()
        if raw_total > 0 and "raw_mean" in mdf.columns:
            grand_mean_raw = (mdf["raw_mean"] * raw_w).sum() / raw_total
            records[f"{metric}__raw__mean_of_mean"] = grand_mean_raw
            if raw_total > 1:
                ss_within = ((raw_w - 1) * mdf["raw_std"] ** 2).sum()
                ss_between = (raw_w * (mdf["raw_mean"] - grand_mean_raw) ** 2).sum()
                records[f"{metric}__raw__pooled_std"] = np.sqrt(
                    (ss_within + ss_between) / (raw_total - 1)
                )
            else:
                records[f"{metric}__raw__pooled_std"] = np.nan

        # --- Normalised variant ---
        norm_w = mdf["norm_count"].fillna(0) if "norm_count" in mdf.columns else raw_w
        norm_total = norm_w.sum()
        if norm_total > 0 and "norm_mean" in mdf.columns:
            grand_mean_norm = (mdf["norm_mean"] * norm_w).sum() / norm_total
            records[f"{metric}__norm__mean_of_mean"] = grand_mean_norm
            if norm_total > 1:
                ss_within = ((norm_w - 1) * mdf["norm_std"] ** 2).sum()
                ss_between = (norm_w * (mdf["norm_mean"] - grand_mean_norm) ** 2).sum()
                records[f"{metric}__norm__pooled_std"] = np.sqrt(
                    (ss_within + ss_between) / (norm_total - 1)
                )
            else:
                records[f"{metric}__norm__pooled_std"] = np.nan

    return pd.Series(records)


def load_dataset(csv_path: Path, pose_records_dir: Path) -> pd.DataFrame:
    """Load all subjects and merge pose metrics with clinical metadata.

    Steps:
    1. Read the clinical CSV.
    2. Filter to rows with valid results_path and ASD/TD diagnosis.
    3. For each subject, load segment-level metrics and aggregate.
    4. Merge with clinical columns.

    Args:
        csv_path: Path to child_for_humanlisbet_paper_with_paths.csv.
        pose_records_dir: Path to the pose_records directory.

    Returns:
        Wide DataFrame: one row per subject, clinical metadata + all metric columns.
    """
    meta = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {len(meta)} rows total")

    # Keep only rows with essential fields
    n_before = len(meta)
    meta = meta.dropna(subset=["results_path"])
    meta = meta[meta["diagnosis"].isin(["ASD", "TD"])]
    logger.info(
        f"After filtering: {len(meta)}/{n_before} rows "
        f"(valid results_path + ASD/TD diagnosis)"
    )

    rows: list[dict] = []
    skipped: list[str] = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading subjects"):
        stem = Path(row["results_path"]).stem
        metrics = load_subject_metrics(pose_records_dir, stem)

        if metrics is None:
            skipped.append(stem)
            continue

        # Keep only defined clinical columns that exist in this CSV
        clinical = {
            k: row[k] for k in CLINICAL_COLS if k in row.index
        }
        clinical.update(metrics.to_dict())
        rows.append(clinical)

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} subjects (missing pose records): "
            + ", ".join(skipped[:5])
            + ("…" if len(skipped) > 5 else "")
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"Dataset built: {len(df)} subjects × "
        f"{len(get_metric_columns(df))} metric columns"
    )
    return df
