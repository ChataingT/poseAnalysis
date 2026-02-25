"""
pose_analysis: Exploratory correlation analysis between pose metrics and child clinical features.

Usage:
    python run_analysis.py \\
        --csv /path/to/child_for_humanlisbet_paper_with_paths.csv \\
        --pose-records /path/to/pose_records/ \\
        --output-dir /path/to/results/
"""

from .load import load_dataset, get_metric_columns, parse_metric_column
from .stats import detect_outliers, run_all_statistics
from .viz import generate_all_figures

__all__ = [
    "load_dataset",
    "get_metric_columns",
    "parse_metric_column",
    "detect_outliers",
    "run_all_statistics",
    "generate_all_figures",
]
