"""
Plot all EXPONENTIAL functions selected in the integrated EXP-LINEAR analysis.

The integrated analysis compares CONSTANT, LINEAR, EXPONENTIAL, and IGNORE.

EXP-chosen families are identified from the model-selection output table.
Their EXPONENTIAL functions are then plotted across each family's observed
chromosome-number range. Positive and negative effective slopes are shown
using the shared slope-direction colors.

For corrected plots:
    exp_p1_tree_scale_corrected is used for the rate scale;
    exp_slope_p2 is unchanged;
    exp_effective_slope_tree_scale_corrected is used for slope direction.

Expected slope-table columns:
    family_name
    exp_p1
    exp_p1_tree_scale_corrected
    exp_slope_p2
    exp_effective_slope
    exp_effective_slope_tree_scale_corrected
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
from source_code.analysis.parameter_utils import (
    CORRECTED_PARAMETER_SCALE,
    RAW_PARAMETER_SCALE,
    VALID_PARAMETER_SCALES,
)
from source_code.analysis.plot_utils import (
    compute_exp_rate,
    ensure_dir,
    plot_direction_colored_function_overlay,
)
from source_code.constants import (
    FAMILY_NAME_COL,
    LABEL_EXP,
    MAX_CHROM_COL,
    MIN_CHROM_COL,
)
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
        help=(
            "Model-selection CSV containing the chosen function "
            "for each family."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the plot.",
    )

    parser.add_argument(
        "--parameter-scale",
        required=True,
        choices=VALID_PARAMETER_SCALES,
        help=(
            "Use raw or tree-scale-corrected EXP rate parameters. "
            "EXP p2 is unchanged by tree-scale correction."
        ),
    )

    parser.add_argument(
        "--log-note",
        type=str,
        default="",
        help="Optional note describing this analysis run.",
    )

    return parser.parse_args()


def resolve_exp_plot_columns(
    parameter_scale: str,
) -> tuple[str, str]:
    """
    Resolve EXP p1 and effective-slope columns for the requested scale.

    EXP p2 is a shape parameter and is therefore not tree-scale corrected.
    """
    if parameter_scale == CORRECTED_PARAMETER_SCALE:
        return (
            f"{EXP_P1_COL}_tree_scale_corrected",
            f"{EXP_EFFECTIVE_SLOPE_COL}_tree_scale_corrected",
        )

    if parameter_scale == RAW_PARAMETER_SCALE:
        return (
            EXP_P1_COL,
            EXP_EFFECTIVE_SLOPE_COL,
        )

    raise ValueError(
        f"Unsupported parameter scale: {parameter_scale}"
    )


def load_exponential_chosen_families(
    slope_plotting_table_file: Path,
    chosen_model_file: Path,
    parameter_scale: str,
) -> tuple[pd.DataFrame, str, str]:

    p1_col, effective_slope_col = resolve_exp_plot_columns(
        parameter_scale
    )

    slope_columns: list[str] = [
        FAMILY_NAME_COL,
        p1_col,
        EXP_P2_COL,
        effective_slope_col,
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
        p1_col,
        EXP_P2_COL,
        effective_slope_col,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=numeric_columns)

    df = df[
        df[effective_slope_col] != 0
    ].copy()

    if df.empty:
        raise ValueError(
            "No plottable EXPONENTIAL-chosen families were found after "
            f"merging {chosen_model_file} with "
            f"{slope_plotting_table_file}."
        )

    return (
        df.sort_values(FAMILY_NAME_COL).reset_index(drop=True),
        p1_col,
        effective_slope_col,
    )


def run_exponential_overlay(
    transition_label: str,
    slope_plotting_table_file: Path,
    chosen_model_file: Path,
    output_dir: Path,
    parameter_scale: str,
) -> tuple[Path, int, str, str]:

    ensure_dir(output_dir)

    (
        exp_df,
        p1_col,
        effective_slope_col,
    ) = load_exponential_chosen_families(
        slope_plotting_table_file=slope_plotting_table_file,
        chosen_model_file=chosen_model_file,
        parameter_scale=parameter_scale,
    )

    output_file: Path = (
        output_dir
        / f"{transition_label}_chosen_exponential_functions_overlay.png"
    )

    plot_direction_colored_function_overlay(
        df=exp_df,
        family_col=FAMILY_NAME_COL,
        p1_col=p1_col,
        p2_col=EXP_P2_COL,
        min_chrom_col=MIN_CHROM_COL,
        max_chrom_col=MAX_CHROM_COL,
        rate_function=compute_exp_rate,
        direction_col=effective_slope_col,
        title=(
            f"{transition_label}: EXPONENTIAL functions selected "
            f"in the integrated analysis ({parameter_scale})"
        ),
        ylabel="Inferred transition rate",
        out_file=output_file,
    )

    return (
        output_file,
        len(exp_df),
        p1_col,
        effective_slope_col,
    )


def main() -> None:
    args: argparse.Namespace = parse_args()

    (
        output_file,
        number_of_families,
        p1_col,
        effective_slope_col,
    ) = run_exponential_overlay(
        transition_label=args.transition_label,
        slope_plotting_table_file=args.slope_plotting_table_file,
        chosen_model_file=args.chosen_model_file,
        output_dir=args.output_dir,
        parameter_scale=args.parameter_scale,
    )

    log_run(
        step="chosen_exponential_functions_overlay",
        script=Path(__file__),
        params={
            "transition_label": args.transition_label,
            "slope_plotting_table_file": (
                args.slope_plotting_table_file
            ),
            "chosen_model_file": args.chosen_model_file,
            "output_dir": args.output_dir,
            "parameter_scale": args.parameter_scale,
            "exp_p1_column": p1_col,
            "exp_p2_column": EXP_P2_COL,
            "effective_slope_column": effective_slope_col,
            "number_of_exponential_chosen_families": (
                number_of_families
            ),
        },
        outputs=[output_file.as_posix()],
        description=(
            "Plotted all EXPONENTIAL functions selected in the integrated "
            "CONSTANT-LINEAR-EXPONENTIAL-IGNORE analysis."
        ),
        notes=(
            "EXP-chosen families were identified from the model-selection "
            "table. Each function was plotted across its family's observed "
            "chromosome-number range and colored by effective-slope "
            f"direction. Parameter scale: {args.parameter_scale}."
            + (
                f"\nRun note: {args.log_note}"
                if args.log_note
                else ""
            )
        ),
    )

    print("[✓] EXPONENTIAL function overlay completed.")
    print(f"Parameter scale: {args.parameter_scale}")
    print(f"EXP p1 column: {p1_col}")
    print(f"EXP p2 column: {EXP_P2_COL}")
    print(
        f"EXP effective slope column: "
        f"{effective_slope_col}"
    )
    print(f"Families plotted: {number_of_families}")
    print(output_file)


if __name__ == "__main__":
    main()