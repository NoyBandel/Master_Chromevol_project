"""
Plot all LINEAR functions selected in the baseline analysis on one graph.

The baseline analysis compares CONSTANT, LINEAR, and IGNORE models.

Each selected LINEAR function is plotted across the observed chromosome-number
range of its family. Positive and negative slopes are shown using the shared
slope-direction colors.

Expected input columns:
    family_name
    chosen_function_label
    lin_p1
    lin_slope_p2
    min_chrom
    max_chrom
"""

import argparse
from pathlib import Path

import pandas as pd

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL, LABEL_LINEAR, LIN_P1_COL, LIN_SLOPE_P2_COL
from source_code.analysis.plot_utils import compute_linear_rate, ensure_dir, plot_direction_colored_function_overlay
from source_code.constants import FAMILY_NAME_COL, MAX_CHROM_COL, MIN_CHROM_COL
from source_code.logger import log_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all baseline LINEAR-chosen functions on one graph.")
    parser.add_argument("--transition-label", required=True, help="Transition label: gain, loss, or dupl.")
    parser.add_argument("--slope-analysis-file", type=Path, required=True, help="Baseline slope-analysis CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which to save the plot.")
    return parser.parse_args()


def load_linear_chosen_families(slope_analysis_file: Path) -> pd.DataFrame:
    required_columns: list[str] = [
        FAMILY_NAME_COL,
        CHOSEN_FUNCTION_LABEL_COL,
        LIN_P1_COL,
        LIN_SLOPE_P2_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
    ]

    df: pd.DataFrame = pd.read_csv(slope_analysis_file, usecols=required_columns)
    df = df[df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_LINEAR].copy()

    numeric_columns: list[str] = [LIN_P1_COL, LIN_SLOPE_P2_COL, MIN_CHROM_COL, MAX_CHROM_COL]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns)
    df = df[df[LIN_SLOPE_P2_COL] != 0].copy()

    if df.empty:
        raise ValueError(f"No plottable LINEAR-chosen families were found in: {slope_analysis_file}")

    return df.sort_values(FAMILY_NAME_COL).reset_index(drop=True)


def run_baseline_linear_overlay(
    transition_label: str,
    slope_analysis_file: Path,
    output_dir: Path,
) -> tuple[Path, int]:
    ensure_dir(output_dir)

    linear_df: pd.DataFrame = load_linear_chosen_families(slope_analysis_file)
    output_file: Path = output_dir / f"{transition_label}_chosen_linear_functions_overlay.png"

    plot_direction_colored_function_overlay(
        df=linear_df,
        family_col=FAMILY_NAME_COL,
        p1_col=LIN_P1_COL,
        p2_col=LIN_SLOPE_P2_COL,
        min_chrom_col=MIN_CHROM_COL,
        max_chrom_col=MAX_CHROM_COL,
        rate_function=compute_linear_rate,
        direction_col=LIN_SLOPE_P2_COL,
        title=f"{transition_label}: LINEAR functions selected in the baseline analysis",
        ylabel="Inferred transition rate",
        out_file=output_file,
    )

    return output_file, len(linear_df)


def main() -> None:
    args: argparse.Namespace = parse_args()

    output_file, number_of_families = run_baseline_linear_overlay(
        transition_label=args.transition_label,
        slope_analysis_file=args.slope_analysis_file,
        output_dir=args.output_dir,
    )

    log_run(
        step="baseline_chosen_linear_functions_overlay",
        script=Path(__file__),
        params={
            "transition_label": args.transition_label,
            "slope_analysis_file": args.slope_analysis_file,
            "output_dir": args.output_dir,
            "number_of_linear_chosen_families": number_of_families,
        },
        outputs=[output_file.as_posix()],
        description="Plotted all LINEAR functions selected in the baseline CONSTANT-LINEAR-IGNORE analysis.",
        notes="Each function was plotted across its family's observed chromosome-number range and colored by slope direction.",
    )

    print("[✓] Baseline LINEAR function overlay completed.")
    print(f"Families plotted: {number_of_families}")
    print(output_file)


if __name__ == "__main__":
    main()