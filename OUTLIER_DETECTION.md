# Outlier detection — detailed documentation

## Why detect outliers in pose metrics?

Pose metrics are derived from automatic skeleton tracking. Even after the `poseToRecord` filtering pipeline (confidence threshold, dyadic filter, continuity filter), the **aggregated video-level values** can still be extreme due to:

- **Tracking failures** — the pose estimator briefly assigns wildly incorrect keypoint positions
- **Occlusion artifacts** — a person stepping briefly out of frame causes spurious high-speed or zero-speed events
- **Normalization edge cases** — if trunk height is poorly estimated for a segment, normalised metrics inherit the error
- **Genuine biological extremes** — a child with very high activity level may produce legitimately large values (these are not errors and should be investigated before excluding)

Outlier detection is applied **after** aggregation (on the video-level `mean_of_mean` and `pooled_std` values), not on individual frames.

---

## Two detection methods used in combination

### Method 1 — Z-score

```
z(x) = |x − mean(X)| / std(X)
Flag if z > threshold  (default: 3.0)
```

A z-score of 3 means the value is more than 3 standard deviations from the group mean. Under a normal distribution this covers 99.73% of values, so anything beyond it is statistically extreme.

**Strength:** intuitive, well-calibrated for symmetric/bell-shaped distributions.

**Weakness:** the mean and std are themselves pulled toward the outlier (called "masking"), so a very extreme value slightly reduces its own z-score. In small samples (< 30) this effect can be noticeable.

### Method 2 — IQR (interquartile range)

```
IQR = Q3 − Q1
Flag if x < Q1 − multiplier × IQR   or   x > Q3 + multiplier × IQR
(default multiplier: 3.0)
```

The IQR spans the central 50% of the data. Because Q1 and Q3 are **rank-based**, they are not distorted by extreme values — making this method robust to the masking problem.

Note: the standard "mild outlier" fence uses 1.5×IQR. We use **3.0×IQR** to flag only the most extreme values, equivalent to the "far outlier" Tukey fence. This is intentionally conservative to avoid over-exclusion.

**Strength:** robust to non-normal and skewed distributions (most speed/distance metrics are right-skewed since they are non-negative).

**Weakness:** can over-flag on bimodal distributions (e.g. if ASD and TD have very different means, the combined IQR is wide and fewer values get flagged than expected).

### Combined rule

A value is flagged if **either** method detects it:

```
outlier = (z_score > z_threshold)  OR  (iqr_method flagged)
```

This union gives broader coverage. The `outlier_report.csv` records which method(s) triggered each flag, so you can assess confidence:

| `z_outlier` | `iqr_outlier` | Interpretation |
|:-----------:|:-------------:|----------------|
| True | True | High confidence — both methods agree this is extreme |
| True | False | z-score driven; possible if distribution is close to normal but this value is far out |
| False | True | IQR driven; typical when the distribution is skewed (heavy right tail) — value is in the far tail but z-score is moderated by a large std |

---

## What happens to detected outliers

Flagged values are **replaced with NaN** for the statistical tests only. The original `merged_dataset.csv` retains the true values.

This means:
- Each statistical test uses a **different effective sample size** per metric (reported as `n_total_used` in the stats CSVs)
- Subjects are never excluded entirely — they contribute to all metrics where their values are not flagged
- The outlier report is saved so you can trace which values were removed

---

## Output files

### `outliers/outlier_report.csv`

One row per flagged (subject × metric) pair.

| Column | Description |
|--------|-------------|
| `subject_uuid` | Session identifier (matches `uuid` in the clinical CSV) |
| `diagnosis` | ASD or TD |
| `metric_col` | Full column name, e.g. `child_speed_centroid__raw__mean_of_mean` |
| `base_metric` | Base metric name, e.g. `child_speed_centroid` |
| `variant` | `raw` or `norm` |
| `stat_type` | `mean_of_mean` or `pooled_std` |
| `value` | The actual flagged value |
| `z_score` | Absolute z-score of this value within its metric column |
| `z_outlier` | True if `z_score > z_threshold` |
| `iqr_outlier` | True if outside the Q1/Q3 ± 3×IQR fence |

**How to use it:**
- Sort by `z_score` descending to find the most extreme individual values
- Filter by `base_metric` to see which metrics have the most problematic data
- Filter by `subject_uuid` to audit a specific subject
- Cross-reference with the subject's HTML quality report in `pose_records/<subject>/report_*.html` for visual confirmation

### `outliers/subject_summary.csv`

One row per subject.

| Column | Description |
|--------|-------------|
| `subject_uuid` | Session identifier |
| `diagnosis` | ASD or TD |
| `n_outlier_metrics` | Number of metric columns flagged for this subject |

**How to interpret:**

| `n_outlier_metrics` | Likely meaning |
|---------------------|----------------|
| 0–2 | Normal — isolated extreme values, no concern |
| 3–8 | Mild — worth checking the HTML report |
| 9–20 | Moderate — tracking quality may be degraded in part of the session |
| > 20 | High — consider whether this recording should be included in the analysis at all |

A subject with many flagged metrics often has a specific recording quality problem (e.g. the child moved out of frame for a sustained period, or a segment had unusually poor lighting). The `pct_valid_raw` column in the segment `metrics_summary.csv` files can confirm this.

### `outliers/metric_summary.csv`

One row per metric column.

| Column | Description |
|--------|-------------|
| `metric_col` | Full column name |
| `n_outliers` | Number of subjects flagged for this metric |
| `pct_outliers` | Percentage of subjects flagged |

**How to interpret:**

| `pct_outliers` | Likely meaning |
|----------------|----------------|
| < 5% | Expected — natural tails of the distribution |
| 5–15% | Elevated — the metric may have a naturally skewed distribution; check the IQR boxplot |
| > 15% | High — possible systematic issue with how this metric is computed, or a bimodal distribution (ASD and TD have very different means) |

> **Important:** a high `pct_outliers` for a metric does **not** necessarily mean the metric is bad. Metrics like `child_speed_kp_left_ankle` naturally have more extreme values than trunk-based metrics because ankle tracking is less reliable. Check the figures in `figures/outliers/` to distinguish genuine extremes from distribution artifacts.

---

## Figures

### `figures/outliers/outlier_boxplot_{variant}_{group}.png`

Box plots of each metric group (social, child motion, child keypoints, clinician) for one variant (`raw` or `norm`).

- The **box** shows Q1, median, Q3
- **Whiskers** extend to 1.5×IQR (standard matplotlib default)
- **Red dots** are subjects flagged as outliers by the combined detection rule

Reading the plot:
- A dot well above/below the whiskers confirms the value is extreme
- Many red dots on the same metric → high `pct_outliers` → check `metric_summary.csv`
- A red dot close to the whisker tip may be a borderline case (IQR-only flag)

### `figures/outliers/per_subject_outlier_count_{variant}.png`

Bar chart of subjects ranked by their `n_outlier_metrics` count, coloured by diagnosis.

This gives an immediate visual overview of whether outlier-prone subjects cluster in one diagnostic group (which would suggest a confound) or are spread randomly (which suggests random recording issues).

---

## Common patterns and what they mean

### Pattern 1: One subject with a very high outlier count

The subject likely had a recording quality problem (tracking loss, occlusion, camera movement). Check their HTML report. Consider excluding this subject from all analyses and re-running — if results change substantially, note this as a sensitivity analysis.

### Pattern 2: A small set of metrics consistently flagged across many subjects

The metric's distribution is likely right-skewed (e.g. keypoint speeds have a long right tail because of occasional large tracking jumps). The IQR method flags the far tail. This is not a bug — it means the "typical" value is near zero but occasional large values inflate the mean. The `pooled_std` aggregation may be more informative for such metrics.

### Pattern 3: Only `iqr_outlier=True` (not `z_outlier`)

The metric distribution is skewed. The mean and std are pulled toward the extreme values, which moderates the z-score. The IQR method is more appropriate here and its flag should be trusted.

### Pattern 4: Only `z_outlier=True` (not `iqr_outlier`)

The distribution is approximately symmetric (or has small IQR). The value is extreme relative to the mean but not beyond the IQR fence. This is the most common type for normally distributed metrics like `facingness` or `congruent_motion`.

### Pattern 5: Outliers concentrated in one diagnosis group

If ASD subjects have systematically more flagged values for a specific metric, it could mean:
1. The metric is genuinely more variable in ASD (interesting finding)
2. Tracking is harder in ASD children (e.g. more movement, more occlusion)

Check the `pct_valid_raw` column in the segment files and the `kp_coverage_after.csv` quality report for that subject to distinguish the two explanations.

---

## Adjusting detection sensitivity

The detection thresholds can be changed at runtime:

```bash
# More permissive (only the most extreme values)
python -m pose_analysis.run_analysis ... --z-threshold 4.0 --iqr-mult 4.0

# Stricter (flag more values)
python -m pose_analysis.run_analysis ... --z-threshold 2.5 --iqr-mult 2.0
```

There is no universally correct threshold. With ~120 subjects and ~240 metric columns:
- At z=3.0 you expect ~0.27% false flags under normality → ~0.003 × 120 × 240 ≈ 86 expected false positives across all metrics
- At z=2.5 this rises to ~1.24% → ~357 expected false positives

For exploratory analysis the defaults (z=3.0, IQR×3.0) are appropriate. For a paper, it is worth reporting sensitivity to this choice.
