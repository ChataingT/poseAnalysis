# pose_analysis

Exploratory correlation analysis between pose-derived behavioural metrics and child clinical features (diagnosis, gender, age, ADOS-2 total score).

---

## Table of contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Inputs](#inputs)
5. [Usage](#usage)
6. [Outputs](#outputs)
7. [Metrics reference](#metrics-reference)
8. [Statistical methods](#statistical-methods)
9. [Module reference](#module-reference)

---

## Overview

The module reads pre-processed pose records (output of `poseToRecord`) for each subject, aggregates frame-level metrics into video-level summaries, and systematically tests for associations with four clinical features:

| Feature | Type | Test used |
|---------|------|-----------|
| `diagnosis` (ASD vs TD) | Binary | Mann-Whitney U + Cohen's d |
| `gender` (Male vs Female) | Binary | Mann-Whitney U + Cohen's d |
| `Ados_2_Age` | Continuous | Spearman correlation |
| `ADOS_2_TOTAL` | Continuous | Spearman correlation |

All tests are corrected for multiple comparisons using Benjamini-Hochberg FDR.

Before running statistics, outlier values are detected per metric (z-score and IQR methods) and replaced with NaN, so each test uses only clean data. The number of subjects actually used is reported alongside each result.

---

## Dependencies

All dependencies are available in the project virtualenv.

| Package | Role |
|---------|------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scipy` | Mann-Whitney U, Spearman correlation, z-score |
| `statsmodels` | Benjamini-Hochberg FDR correction |
| `matplotlib` | Figure rendering |
| `seaborn` | Violin plots and heatmaps |
| `tqdm` | Progress bars during data loading |

---

## Installation

No installation step is required beyond activating the project environment:

```bash
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate
```

The module is used **in-place** from the `humanLISBET-paper/` directory, imported as a Python package:

```bash
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper
python -m pose_analysis.run_analysis --help
```

---

## Inputs

### 1. Clinical metadata CSV

**Path (default):**
```
humanLISBET-paper/dataset/info/child_for_humanlisbet_paper_with_paths.csv
```

One row per subject. Relevant columns:

| Column | Description |
|--------|-------------|
| `uuid` | Unique session identifier (e.g. `7772_Visite2_Recherche`) |
| `diagnosis` | `ASD` or `TD` (typical development) |
| `gender` | `Male` or `Female` |
| `Ados_2_Age` | Age at ADOS-2 assessment in decimal years |
| `ADOS_2_TOTAL` | ADOS-2 total score (higher = more severe) |
| `results_path` | Absolute path to the source pose JSON file — its stem maps to the pose_records subdirectory |

Rows missing `results_path` or with a diagnosis other than `ASD`/`TD` are automatically skipped.

### 2. Pose records directory

**Path (default):**
```
humanLISBET-paper/dataset/pose_records/
```

One subdirectory per subject, named after the stem of `results_path`:

```
pose_records/
└── results_skeleton_7772_T2a_ADOS/
    └── segments/
        ├── seg_001/
        │   └── metrics_summary.csv
        ├── seg_002/
        │   └── metrics_summary.csv
        └── ...
```

Each `metrics_summary.csv` contains one row per metric with columns:

| Column | Description |
|--------|-------------|
| `raw_count` | Number of valid frames in this segment (raw) |
| `raw_mean` | Mean metric value over valid frames (raw, pixel units) |
| `raw_std` | Standard deviation over valid frames (raw) |
| `raw_min/25%/50%/75%/max` | Raw percentiles |
| `norm_count` | Number of valid frames (normalised) |
| `norm_mean` | Mean metric value (normalised by trunk height) |
| `norm_std` | Standard deviation (normalised) |
| `norm_min/25%/50%/75%/max` | Normalised percentiles |
| `pct_valid_raw` | Percentage of frames where the metric was computable |

---

## Usage

### Interactive run

```bash
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

python -m pose_analysis.run_analysis \
    --csv         dataset/info/child_for_humanlisbet_paper_with_paths.csv \
    --pose-records dataset/pose_records \
    --output-dir  pose_analysis/results
```

### SLURM submission

```bash
sbatch pose_analysis/run_analysis.slurm
```

Logs are written to `pose_analysis/logs/pose_analysis_<jobid>.out/.err`.

### All CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | *(required)* | Path to the clinical metadata CSV |
| `--pose-records` | *(required)* | Path to the `pose_records/` directory |
| `--output-dir` | `./results` | Root directory for all outputs |
| `--alpha` | `0.05` | FDR significance threshold |
| `--z-threshold` | `3.0` | Z-score threshold for outlier detection |
| `--iqr-mult` | `3.0` | IQR multiplier for outlier detection |
| `--min-subjects` | `5` | Minimum subjects per group to run a test |
| `--max-individual-plots` | `150` | Cap on per-metric violin/scatter plots |
| `--skip-figures` | off | Run statistics only, skip figure generation |
| `--log-level` | `INFO` | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

To add new subjects, simply add rows to the CSV (with valid `results_path`) and rerun the script — it rebuilds everything from scratch.

---

## Outputs

```
results/
├── merged_dataset.csv
├── outliers/
│   ├── outlier_report.csv
│   ├── subject_summary.csv
│   └── metric_summary.csv
├── stats/
│   ├── binary_diagnosis_raw.csv
│   ├── binary_diagnosis_norm.csv
│   ├── binary_gender_raw.csv
│   ├── binary_gender_norm.csv
│   ├── continuous_age_raw.csv
│   ├── continuous_age_norm.csv
│   ├── continuous_ados_total_raw.csv
│   ├── continuous_ados_total_norm.csv
│   └── top_significant_metrics.csv
└── figures/
    ├── heatmap_{raw|norm}_{mean_of_mean|pooled_std}.png
    ├── top20_{feature}_{raw|norm}_{stat_type}.png
    ├── volcano_{feature}_{raw|norm}_{stat_type}.png
    ├── social_panel_{raw|norm}_{stat_type}.png
    ├── outliers/
    │   ├── outlier_boxplot_{raw|norm}_{group}.png
    │   └── per_subject_outlier_count_{raw|norm}.png
    ├── violin/{feature}/{base_metric}_{raw|norm}.png
    └── scatter/{feature}/{base_metric}_{raw|norm}.png
```

### `merged_dataset.csv`

Wide table with one row per subject. Columns:
- Clinical metadata columns (uuid, diagnosis, gender, age, ADOS score, …)
- One column per aggregated metric, named `{base_metric}__{raw|norm}__{stat_type}`

The aggregation converts multiple segments into two summary statistics per metric:
- `mean_of_mean` — weighted average of segment means (weight = number of valid frames). This is the primary metric value.
- `pooled_std` — combined standard deviation across all segments using the exact pooled-variance formula: `sqrt((Σ(n_i−1)·s_i² + Σ n_i·(μ_i−μ̄)²) / (N−1))`. This captures **total variability** — both within-segment fluctuations and between-segment differences in average behaviour.

### `outliers/outlier_report.csv`

Long-form table of every flagged value. Columns:

| Column | Description |
|--------|-------------|
| `subject_uuid` | Subject identifier |
| `diagnosis` | ASD or TD |
| `metric_col` | Full metric column name |
| `base_metric` | Base metric name (e.g. `child_speed_centroid`) |
| `variant` | `raw` or `norm` |
| `stat_type` | `mean_of_mean` or `pooled_std` |
| `value` | The flagged value |
| `z_score` | Absolute z-score |
| `z_outlier` | True if flagged by z-score method |
| `iqr_outlier` | True if flagged by IQR method |

### `outliers/subject_summary.csv`

Per-subject count of how many metrics were flagged. Useful for identifying subjects with systematically aberrant recordings.

### `outliers/metric_summary.csv`

Per-metric count and percentage of subjects flagged. Useful for identifying metrics with poor quality or high natural variance.

### `stats/binary_{feature}_{variant}.csv`

One row per metric (both `mean_of_mean` and `pooled_std` aggregations). Columns:

| Column | Description |
|--------|-------------|
| `metric_col` | Full column name in `merged_dataset.csv` |
| `base_metric` | Base metric name |
| `variant` | `raw` or `norm` |
| `stat_type` | `mean_of_mean` or `pooled_std` |
| `n_ASD` / `n_TD` | Number of subjects used per group |
| `mean_ASD`, `std_ASD`, `median_ASD` | Group descriptive stats |
| `mean_TD`, `std_TD`, `median_TD` | Group descriptive stats |
| `n_total_used` | Total subjects used (after outlier removal) |
| `mwu_statistic` | Mann-Whitney U statistic |
| `pvalue` | Uncorrected p-value |
| `cohens_d` | Cohen's d effect size (positive = group1 > group2) |
| `pvalue_fdr` | FDR-corrected p-value (Benjamini-Hochberg) |
| `significant_fdr` | True if `pvalue_fdr < alpha` |

### `stats/continuous_{feature}_{variant}.csv`

Same structure as binary, replacing group columns with:

| Column | Description |
|--------|-------------|
| `n_total_used` | Number of valid paired observations |
| `spearman_rho` | Spearman rank correlation coefficient (−1 to +1) |
| `pvalue` | Uncorrected p-value |
| `pvalue_fdr` | FDR-corrected p-value |
| `significant_fdr` | True if `pvalue_fdr < alpha` |

### `stats/top_significant_metrics.csv`

Unified summary of the top 20 significant metrics per (feature, variant) pair, sorted by absolute effect size. Quick reference for the most informative metrics.

### Figures

| Figure | Description |
|--------|-------------|
| `heatmap_{variant}_{stat_type}.png` | Matrix of signed effect sizes for all metrics × all features. Cells with FDR significance are annotated with stars (*, **, ***). Rows are hierarchically clustered. |
| `top20_{feature}_{variant}_{stat_type}.png` | Horizontal bar chart of the 20 metrics with the largest absolute effect size. Red = positive effect, blue = negative, grey = not significant. |
| `volcano_{feature}_{variant}_{stat_type}.png` | Volcano plot: x = signed effect size, y = −log₁₀(FDR p-value). Top points are labelled. Horizontal dashed line = FDR threshold. |
| `social_panel_{variant}_{stat_type}.png` | Grid of violin plots for the 6 social/dyadic metrics vs diagnosis. |
| `outliers/outlier_boxplot_{variant}_{group}.png` | Box plots of each metric group with outlier subjects highlighted in red. |
| `outliers/per_subject_outlier_count_{variant}.png` | Bar chart of subjects ranked by number of outlier metrics (ASD in red, TD in blue). |
| `violin/{feature}/{base_metric}_{variant}.png` | Per-metric violin + strip plot for binary comparisons. Generated only for metrics with FDR < alpha. |
| `scatter/{feature}/{base_metric}_{variant}.png` | Per-metric scatter plot with regression line for continuous features. Generated only for FDR-significant metrics. |

---

## Metrics reference

All metrics are produced by `poseToRecord` and aggregated here. They come in two variants:

- **raw** — original pixel-space units (speeds in px/frame, distances in px)
- **norm** — divided by the subject's trunk height (a proxy for depth normalisation, making cross-subject comparisons more meaningful). Angles and correlations are unchanged.

### Child individual metrics

| Metric | Description |
|--------|-------------|
| `child_speed_centroid` | Frame-to-frame displacement of the centroid of all visible keypoints |
| `child_speed_trunk` | Frame-to-frame displacement of the trunk centroid (shoulders + hips) |
| `child_velocity_centroid_x/y` | Signed x/y displacement of centroid (direction-aware) |
| `child_velocity_trunk_x/y` | Signed x/y displacement of trunk centroid |
| `child_acceleration_centroid` | Absolute change in centroid speed between consecutive frames |
| `child_acceleration_trunk` | Absolute change in trunk speed between consecutive frames |
| `child_kinetic_energy` | Sum of squared per-keypoint displacements — a robust measure of overall movement energy |
| `child_speed_kp_{keypoint}` | Speed of one specific keypoint (17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |

### Clinician individual metrics

Same set as child, with `clinician_` prefix.

### Dyadic (interaction) metrics

| Metric | Description | ASD hypothesis |
|--------|-------------|----------------|
| `interpersonal_distance_centroid` | Euclidean distance between child and clinician centroids | Higher in ASD (social avoidance) |
| `interpersonal_distance_trunk` | Distance between trunk-only centroids | Higher in ASD |
| `interpersonal_approach` | Frame-to-frame change in distance (negative = approaching) | Less approaching in ASD |
| `facingness` | Cosine similarity of torso heading vectors. +1 = side-by-side, −1 = face-to-face. In a frontal camera view, mutual face-to-face orientation ≈ −1. | Less face-to-face orientation in ASD |
| `congruent_motion` | Pearson correlation of child and clinician speeds in a rolling 60-frame window. Measures behavioural synchrony. | Lower synchrony in ASD |
| `agitation_global_ke` | Mean kinetic energy of both participants. Global measure of session agitation. | Higher in more severe ASD |

---

## Statistical methods

### Aggregation

For each subject, metrics from all video segments are combined with a **weighted mean** (weight = number of valid frames per segment), producing two summary values per metric:
- `mean_of_mean`: the representative value for the whole session
- `pooled_std`: the exact combined standard deviation (within + between segment variance)

### Outlier detection

Applied independently per metric column before any statistical tests:

1. **Z-score method**: values with |z| > threshold (default 3.0) are flagged
2. **IQR method**: values below Q1 − 3·IQR or above Q3 + 3·IQR are flagged
3. A value is flagged if **either** method detects it
4. Flagged values are replaced with NaN for statistical tests (not excluded from the dataset entirely)

### Tests for binary features

**Mann-Whitney U test** (non-parametric, no normality assumption):
- Null hypothesis: the two groups are drawn from the same distribution
- Two-sided

**Cohen's d** (effect size):
- Computed with pooled standard deviation
- Positive values = group1 (ASD or Male) has a higher mean
- Interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large

### Tests for continuous features

**Spearman rank correlation** (non-parametric, robust to outliers and non-linearity):
- ρ ranges from −1 (perfect negative) to +1 (perfect positive)
- Tests monotonic association between the clinical feature and the metric

### Multiple comparison correction

**Benjamini-Hochberg False Discovery Rate (FDR)** is applied separately for each (feature × variant) combination. With ~240 metrics per block, this controls the expected proportion of false positives among significant results.

A result is considered significant when `pvalue_fdr < alpha` (default 0.05).
