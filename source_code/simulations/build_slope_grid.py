from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from source_code.constants import LABEL_LINEAR
from source_code.analysis.analysis_constants import BASELINE_ANALYSIS_DIR, LIN_SLOPE_P2_COL, CHOSEN_FUNCTION_LABEL_COL
from source_code.simulations.simulation_constants import *
from source_code.logger import log_run



def build_slope_grid(slope_analysis_file: Path) -> pd.DataFrame:
    """
    Build a slope grid using fitted slopes from families where the LINEAR
    model was chosen.

    The grid contains:
    - zero_slope = 0
    - min
    - q1_range = min + (max - min) / 4
    - median
    - mean
    - q3_range = min + 3 * (max - min) / 4
    - max
    """
    slope_analysis_df = pd.read_csv(slope_analysis_file)

    required_cols = {CHOSEN_FUNCTION_LABEL_COL, LIN_SLOPE_P2_COL}
    missing_cols = required_cols - set(slope_analysis_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    linear_slopes_df = slope_analysis_df.loc[slope_analysis_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR].copy()
    slopes = pd.to_numeric(linear_slopes_df[LIN_SLOPE_P2_COL], errors="coerce").dropna()
    if slopes.empty:
        raise ValueError(f"No valid LINEAR slopes found in: {slope_analysis_file}")

    min_slope = float(slopes.min())
    max_slope = float(slopes.max())
    slope_range = max_slope - min_slope

    slope_grid_df = pd.DataFrame([
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_ZERO,
            SLOPE_VALUE_COL: 0.0,
        },
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_MIN,
            SLOPE_VALUE_COL: min_slope,
        },
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_Q1_RANGE,
            SLOPE_VALUE_COL: min_slope + slope_range / 4,
        },
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_MEDIAN,
            SLOPE_VALUE_COL: float(slopes.median()),
        },
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_MEAN,
            SLOPE_VALUE_COL: float(slopes.mean()),
        },
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_Q3_RANGE,
            SLOPE_VALUE_COL: min_slope + 3 * slope_range / 4,
        },
        {
            SLOPE_LABEL_COL: SLOPE_LABEL_MAX,
            SLOPE_VALUE_COL: max_slope,
        },
    ])

    return slope_grid_df


def save_slope_grid(slope_grid_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slope_grid_df.to_csv(output_path, index=False)
    print(f"[✓] Slope grid saved: {output_path}")



def parse_args() -> Namespace:
    parser = ArgumentParser(description="Build a slope grid for ChromEvol slope-grid simulations.")
    parser.add_argument("--transition", type=str, required=True, choices=VALID_SIMULATION_TRANSITIONS, help="Transition to build the slope grid for.")
    parser.add_argument("--slope-analysis-file", type=Path, required=True, help="Path to the transition-specific slope analysis table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transition = args.transition
    slope_analysis_file = args.slope_analysis_file
    print("bug")
    output_path = (SIMULATIONS_DIR / transition / SLOPE_GRID_SIMULATIONS_LABEL / SLOPE_GRID_FILENAME)
    slope_grid_df = build_slope_grid(slope_analysis_file=slope_analysis_file)
    save_slope_grid(slope_grid_df=slope_grid_df, output_path=output_path)

    log_run(
        step="slope_grid_simulations",
        script=Path(__file__),
        params={
            "transition": transition,
            "slope_analysis_file": slope_analysis_file,
        },
        outputs=[
            output_path.as_posix(),
        ],
        description="Built a transition-specific slope grid for ChromEvol slope-grid simulations.",
        notes=(
            "Slope values were calculated only from families where the LINEAR model was chosen. "
            "The zero_slope row represents LINEAR p2=0 for simulation design, not a separate "
            "CONST-function simulation. "
            f"Slope grid labels: {', '.join(SLOPE_LABELS_ORDERED)}."
        ),
    )


if __name__ == "__main__":
    main()