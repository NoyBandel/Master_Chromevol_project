"""
Plot all EXPONENTIAL functions selected in the integrated EXP-LINEAR analysis.

The integrated analysis compares CONSTANT, LINEAR, EXPONENTIAL, and IGNORE.

EXP-chosen families are identified from the model-selection output table.
Their EXPONENTIAL functions are then plotted across each family's observed
chromosome-number range. Positive and negative effective slopes are shown
using the shared slope-direction colors.

Expected slope-table columns:
    family_name
    exp_p1
    exp_slope_p2
    exp_effective_slope
    min_chrom
    max_chrom

Expected chosen-model-table columns:
    family_name
    chosen_function_label
"""

import argparse
from pathlib import Path

import pandas as pd

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL
from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import (
    EXP_EFFECTIVE_SLOPE_COL,
    EXP_P1_COL,
    EXP_P2_COL,
)
from source_code.analysis.plot_utils import (
    compute_exp_rate,
    ensure_dir,
    plot_direction_colored_function_overlay,
)
from source_code.constants import FAMILY_NAME_COL, LABEL_EXP, MAX_CHROM_COL, MIN_CHROM_COL
from source_code.logger import log_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all EXPONENTIAL functions selected in the integrated "
            "EXP-LINEAR analysis."
        )
    )
    parser.add_argument(
        "--transition-label",
        required=True,
        help="Transition label: gain, loss, or dupl.",
    )
    parser.add_argument(
        "--slope-plotting-table-file",
        type=Path,
        required=True,
        help="EXP-LINEAR slope plotting table.",
    )
    parser.add_argument(
        "--chosen-model-file",
        type=Path,
        required=True,
        help="Model-selection CSV containing the chosen function for each family.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the plot.",
    )
    return parser.parse_args()


def load_exponential_chosen_families(
    slope_plotting_table_file: Path,
    chosen_model_file: Path,
) -> pd.DataFrame:
    slope_columns: list[str] = [
        FAMILY_NAME_COL,
        EXP_P1_COL,
        EXP_P2_COL,
        EXP_EFFECTIVE_SLOPE_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
    ]

    chosen_columns: list[str] = [
        FAMILY_NAME_COL,
        CHOSEN_FUNCTION_LABEL_COL,
    ]

    slope_df: pd.DataFrame = pd.read_csv(
        slope_plotting_table_file,
        usecols=slope_columns,
    )

    chosen_df: pd.DataFrame = pd.read_csv(
        chosen_model_file,
        usecols=chosen_columns,
    )

    chosen_df = chosen_df[
        chosen_df[CHOSEN_FUNCTION_LABEL_COL] == LABEL_EXP
    ].copy()

    df: pd.DataFrame = chosen_df.merge(
        slope_df,
        on=FAMILY_NAME_COL,
        how="inner",
        validate="one_to_one",
    )

    numeric_columns: list[str] = [
        EXP_P1_COL,
        EXP_P2_COL,
        EXP_EFFECTIVE_SLOPE_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns)
    df = df[df[EXP_EFFECTIVE_SLOPE_COL] != 0].copy()

    if df.empty:
        raise ValueError(
            "No plottable EXPONENTIAL-chosen families were found after "
            f"merging {chosen_model_file} with {slope_plotting_table_file}."
        )

    return df.sort_values(FAMILY_NAME_COL).reset_index(drop=True)


def run_exponential_overlay(
    transition_label: str,
    slope_plotting_table_file: Path,
    chosen_model_file: Path,
    output_dir: Path,
) -> tuple[Path, int]:
    ensure_dir(output_dir)

    exp_df: pd.DataFrame = load_exponential_chosen_families(
        slope_plotting_table_file=slope_plotting_table_file,
        chosen_model_file=chosen_model_file,
    )

    output_file: Path = (
        output_dir
        / f"{transition_label}_chosen_exponential_functions_overlay.png"
    )

    plot_direction_colored_function_overlay(
        df=exp_df,
        family_col=FAMILY_NAME_COL,
        p1_col=EXP_P1_COL,
        p2_col=EXP_P2_COL,
        min_chrom_col=MIN_CHROM_COL,
        max_chrom_col=MAX_CHROM_COL,
        rate_function=compute_exp_rate,
        direction_col=EXP_EFFECTIVE_SLOPE_COL,
        title=(
            f"{transition_label}: EXPONENTIAL functions selected "
            "in the integrated analysis"
        ),
        ylabel="Inferred transition rate",
        out_file=output_file,
    )

    return output_file, len(exp_df)


def main() -> None:
    args: argparse.Namespace = parse_args()

    output_file, number_of_families = run_exponential_overlay(
        transition_label=args.transition_label,
        slope_plotting_table_file=args.slope_plotting_table_file,
        chosen_model_file=args.chosen_model_file,
        output_dir=args.output_dir,
    )

    log_run(
        step="chosen_exponential_functions_overlay",
        script=Path(__file__),
        params={
            "transition_label": args.transition_label,
            "slope_plotting_table_file": args.slope_plotting_table_file,
            "chosen_model_file": args.chosen_model_file,
            "output_dir": args.output_dir,
            "number_of_exponential_chosen_families": number_of_families,
        },
        outputs=[output_file.as_posix()],
        description=(
            "Plotted all EXPONENTIAL functions selected in the integrated "
            "CONSTANT-LINEAR-EXPONENTIAL-IGNORE analysis."
        ),
        notes=(
            "EXP-chosen families were identified from the model-selection "
            "table. Each function was plotted across its family's observed "
            "chromosome-number range and colored by effective-slope direction."
        ),
    )

    print("[✓] EXPONENTIAL function overlay completed.")
    print(f"Families plotted: {number_of_families}")
    print(output_file)


if __name__ == "__main__":
    main()