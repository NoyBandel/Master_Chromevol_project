"""
Plot fitted transition-rate functions separately for every family.

Input is a canonical fitted-functions table created with
fitted_functions_utils.build_fitted_functions_table().

Any registered function type and any number of parameters are supported.
The chosen function is solid; other fitted functions are dashed.
Non-plottable functions such as IGNORE remain in the AICc ranking but are not drawn.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL, MODEL_COLOR_MAP
from source_code.analysis.fitted_functions.fitted_functions_utils import FITTED_FUNCTIONS_TABLE_COLS, FITTED_PARAMETERS_JSON_COL, PARAMETER_SCALE_COL, evaluate_fitted_function, is_plottable_function, parse_parameters
from source_code.analysis.plot_utils import build_chromosome_grid, ensure_dir, safe_name_for_file, save_close, to_float
from source_code.constants import AICC_COL, FAMILY_NAME_COL, LABEL_FUNC_TYPE_COL, LABEL_TESTED_TRANSITION_COL, MAX_CHROM_COL, MIN_CHROM_COL
from source_code.logger import log_run


CHOSEN_LINESTYLE: str = "-"
OTHER_LINESTYLE: str = "--"
CHOSEN_LINEWIDTH: float = 2.8
OTHER_LINEWIDTH: float = 1.7
PLOT_RANGE_PADDING_FRACTION: float = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fitted functions separately for every family.")
    parser.add_argument("--fitted-functions-file", type=Path, required=True, help="Canonical fitted-functions CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which to save per-family plots.")
    parser.add_argument("--log-note", type=str, default="", help="Optional note describing this analysis run.")
    return parser.parse_args()


def load_fitted_functions_table(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Fitted-functions file not found: {file_path}")

    df: pd.DataFrame = pd.read_csv(file_path)
    missing: list[str] = [col for col in FITTED_FUNCTIONS_TABLE_COLS if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns in {file_path}: {missing}")

    if df.duplicated([FAMILY_NAME_COL, LABEL_TESTED_TRANSITION_COL, LABEL_FUNC_TYPE_COL]).any():
        raise ValueError("Duplicate family × transition × function rows found.")

    for col in [AICC_COL, MIN_CHROM_COL, MAX_CHROM_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_single_value(df: pd.DataFrame, col: str, context: str) -> object:
    values: np.ndarray = df[col].dropna().unique()

    if len(values) != 1:
        raise ValueError(f"{context}: expected one '{col}' value, found {values.tolist()}")

    return values[0]


def format_aicc_ranking_text(family_df: pd.DataFrame) -> str:
    ranking_df: pd.DataFrame = family_df[[LABEL_FUNC_TYPE_COL, AICC_COL]].dropna().sort_values(AICC_COL)

    if ranking_df.empty:
        return "AICc ranking\nmissing"

    lines: list[str] = ["AICc ranking"]
    lines.extend(f"{rank}. {row[LABEL_FUNC_TYPE_COL]:<11} {row[AICC_COL]:.3f}" for rank, (_, row) in enumerate(ranking_df.iterrows(), start=1))
    return "\n".join(lines)


def plot_family_functions(family_df: pd.DataFrame, output_dir: Path) -> Path:
    family_name: str = str(get_single_value(family_df, FAMILY_NAME_COL, "Family"))
    transition: str = str(get_single_value(family_df, LABEL_TESTED_TRANSITION_COL, family_name))
    chosen_function: str = str(get_single_value(family_df, CHOSEN_FUNCTION_LABEL_COL, family_name))
    parameter_scale: str = str(get_single_value(family_df, PARAMETER_SCALE_COL, family_name))
    min_chr: float = to_float(get_single_value(family_df, MIN_CHROM_COL, family_name))
    max_chr: float = to_float(get_single_value(family_df, MAX_CHROM_COL, family_name))

    plot_padding: float = (max_chr - min_chr) * PLOT_RANGE_PADDING_FRACTION
    chromosome_grid: np.ndarray | None = build_chromosome_grid(min_chr, max_chr, padding=plot_padding)

    if chromosome_grid is None:
        raise ValueError(f"Could not construct chromosome grid for {family_name}: min={min_chr}, max={max_chr}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(min_chr, max_chr, color="lightgrey", alpha=0.3, zorder=0)

    plotted_functions: int = 0

    for _, row in family_df.iterrows():
        function_label: str = str(row[LABEL_FUNC_TYPE_COL])

        if not is_plottable_function(function_label):
            continue

        parameters: dict[str, float] = parse_parameters(row[FITTED_PARAMETERS_JSON_COL])
        rates: np.ndarray = np.asarray(evaluate_fitted_function(function_label, chromosome_grid, parameters),
                                       dtype=float)

        if not np.isfinite(rates).all():
            raise ValueError(f"Non-finite fitted rates for {family_name}: {function_label}")

        is_chosen: bool = function_label == chosen_function
        legend_label: str = f"{function_label.upper()} (chosen)" if is_chosen else function_label.upper()

        ax.plot(
            chromosome_grid,
            rates,
            label=legend_label,
            color=MODEL_COLOR_MAP.get(function_label),
            linestyle=CHOSEN_LINESTYLE if is_chosen else OTHER_LINESTYLE,
            linewidth=CHOSEN_LINEWIDTH if is_chosen else OTHER_LINEWIDTH,
            alpha=1.0 if is_chosen else 0.8,
        )
        plotted_functions += 1

    if plotted_functions == 0:
        plt.close(fig)
        raise ValueError(f"No registered plottable functions found for {family_name}.")

    ax.set_xlim(float(chromosome_grid.min()), float(chromosome_grid.max()))
    ax.set_ylim(bottom=0)
    ax.set_title(f"{transition}: {family_name}\nChosen function: {chosen_function.upper()} | {parameter_scale}")
    ax.set_xlabel("Chromosome number")
    ax.set_ylabel("Inferred transition rate")
    ax.legend(loc="upper left")

    ax.text(
        0.98,
        0.98,
        format_aicc_ranking_text(family_df),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.92),
    )

    output_file: Path = output_dir / f"{safe_name_for_file(transition)}_{safe_name_for_file(family_name)}_fitted_functions_{parameter_scale}.png"
    save_close(fig, output_file)
    return output_file


def run_per_family_plots(fitted_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    ensure_dir(output_dir)
    return [plot_family_functions(family_df, output_dir) for _, family_df in fitted_df.groupby([LABEL_TESTED_TRANSITION_COL, FAMILY_NAME_COL], sort=True)]


def main() -> None:
    args: argparse.Namespace = parse_args()

    fitted_df: pd.DataFrame = load_fitted_functions_table(args.fitted_functions_file)
    output_files: list[Path] = run_per_family_plots(fitted_df, args.output_dir)

    transitions: list[str] = sorted(fitted_df[LABEL_TESTED_TRANSITION_COL].astype(str).unique().tolist())
    parameter_scales: list[str] = sorted(fitted_df[PARAMETER_SCALE_COL].astype(str).unique().tolist())

    notes: str = "Plotted every registered fitted function for each family. The chosen function was solid and alternatives were dashed. Non-plottable functions were included only in the AICc ranking."
    if args.log_note:
        notes += f"\nRun note: {args.log_note}"

    log_run(
        step="per_family_fitted_functions",
        script=Path(__file__),
        params={
            "fitted_functions_file": str(args.fitted_functions_file),
            "output_dir": str(args.output_dir),
            "transitions": transitions,
            "parameter_scales": parameter_scales,
            "number_of_generated_plots": len(output_files),
        },
        outputs=[output_file.as_posix() for output_file in output_files],
        description="Plotted fitted transition-rate functions separately for every family.",
        notes=notes,
        log_relative_path=Path("fitted_functions") / f"{args.fitted_functions_file.stem}.log",
    )

    print("[✓] Per-family fitted-function plots completed.")
    print(f"Plots generated: {len(output_files)}")
    print(f"Transitions: {transitions}")
    print(f"Parameter scales: {parameter_scales}")
    print(args.output_dir)


if __name__ == "__main__":
    main()