from pathlib import Path
from typing import List, Tuple

from source_code.constants import (
    PROJECT_ROOT,
    LABEL_LOSS,
    LABEL_GAIN,
    LABEL_DUPL,
)


# =========================================================
# SIMULATION ROOTS
# =========================================================

SIMULATIONS_DIR: Path = PROJECT_ROOT / "simulations"

VALID_SIMULATION_TRANSITIONS: Tuple[str, ...] = (
    LABEL_LOSS,
    LABEL_GAIN,
    LABEL_DUPL,
)


# =========================================================
# SLOPE-GRID SIMULATIONS
# =========================================================

SLOPE_GRID_SIMULATIONS_LABEL: str = "slope_grid_simulations"

SLOPE_ANALYSIS_SUBDIR: str = "slope_analysis"
SLOPE_ANALYSIS_TABLE_SUFFIX: str = "slope_analysis_table.csv"

SLOPE_GRID_FILENAME: str = "slope_grid.csv"

RUNS_DIR_NAME: str = "runs"
EXAMPLE_RUNS_DIR_NAME: str = "example_runs"


# =========================================================
# SLOPE-GRID LABELS
# =========================================================

SLOPE_LABEL_ZERO: str = "zero_slope"
SLOPE_LABEL_MIN: str = "min"
SLOPE_LABEL_Q1_RANGE: str = "q1_range"
SLOPE_LABEL_MEDIAN: str = "median"
SLOPE_LABEL_MEAN: str = "mean"
SLOPE_LABEL_Q3_RANGE: str = "q3_range"
SLOPE_LABEL_MAX: str = "max"

SLOPE_LABELS_ORDERED: Tuple[str, ...] = (
    SLOPE_LABEL_ZERO,
    SLOPE_LABEL_MIN,
    SLOPE_LABEL_Q1_RANGE,
    SLOPE_LABEL_MEDIAN,
    SLOPE_LABEL_MEAN,
    SLOPE_LABEL_Q3_RANGE,
    SLOPE_LABEL_MAX,
)


# =========================================================
# SLOPE-GRID COLUMNS
# =========================================================

SLOPE_LABEL_COL: str = "slope_label"
SLOPE_VALUE_COL: str = "slope_value"

SLOPE_GRID_COLS: List[str] = [
    SLOPE_LABEL_COL,
    SLOPE_VALUE_COL,
]