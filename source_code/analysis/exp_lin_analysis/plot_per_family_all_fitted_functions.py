"""
Plot all fitted transition-rate functions separately for every family.

Each plot contains the fitted CONSTANT, LINEAR, and EXPONENTIAL functions
across the family's observed chromosome-number range.

The model selected by the integrated CONSTANT-LINEAR-EXPONENTIAL-IGNORE
comparison is plotted with a solid line. Other fitted functions are dashed.
When IGNORE is selected, all three fitted functions are dashed.

All rates use tree-scale-corrected parameters.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source_code.analysis.analysis_constants import CHOSEN_FUNCTION_LABEL_COL, MODEL_COLOR_MAP
from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import AICC_COL_BY_MODEL, CONSTANT_AICC_COL, EXP_AICC_COL, EXP_P2_COL, IGNORE_AICC_COL, LINEAR_AICC_COL, MODEL_LABELS_WITH_EXP
from source_code.analysis.plot_utils import build_chromosome_grid, compute_constant_rate, compute_exp_rate, compute_linear_rate, ensure_dir, save_close, to_float
from source_code.constants import FAMILY_NAME_COL, LABEL_CONSTANT, LABEL_EXP, LABEL_IGNORE, LABEL_LINEAR, MAX_CHROM_COL, MIN_CHROM_COL
from source_code.logger import log_run


TREE_SCALING_FACTOR_COL: str = "tree_scaling_factor"
CONSTANT_RATE_COL: str = "constant_value_tree_scale_corrected"
LINEAR_P1_COL: str = "lin_p1_tree_scale_corrected"
LINEAR_P2_COL: str = "lin_slope_p2_tree_scale_corrected"
EXP_P1_COL: str = "exp_p1_tree_scale_corrected"

CHOSEN_LINESTYLE: str = "-"
OTHER_LINESTYLE: str = "--"
CHOSEN_LINEWIDTH: float = 2.8
OTHER_LINEWIDTH: float = 1.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all fitted functions separately for every family.")
    parser.add_argument("--transition-label", required=True, help="Transition label: gain, loss, or dupl.")
    parser.add_argument("--chosen-model-file", type=Path, required=True, help="Integrated four-model selection table.")
    parser.add_argument("--fitted-parameters-file", type=Path, required=True, help="Complete fitted-function parameter table.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which to save the per-family plots.")
    return parser.parse_args()


def validate_unique_families(df: pd.DataFrame, file_path: Path) -> None:
    duplicate_mask: pd.Series = df[FAMILY_NAME_COL].duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_families: list[str] = sorted(df.loc[duplicate_mask, FAMILY_NAME_COL].astype(str).unique().tolist())
        raise ValueError(f"Duplicate families in {file_path}: {duplicate_families}")


def load_plotting_table(chosen_model_file: Path, fitted_parameters_file: Path) -> pd.DataFrame:
    parameter_columns: list[str] = [
        FAMILY_NAME_COL,
        TREE_SCALING_FACTOR_COL,
        CONSTANT_RATE_COL,
        LINEAR_P1_COL,
        LINEAR_P2_COL,
        EXP_P1_COL,
        EXP_P2_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
        CONSTANT_AICC_COL,
        LINEAR_AICC_COL,
        EXP_AICC_COL,
        IGNORE_AICC_COL,
    ]
    chosen_columns: list[str] = [FAMILY_NAME_COL, CHOSEN_FUNCTION_LABEL_COL]

    parameter_df: pd.DataFrame = pd.read_csv(fitted_parameters_file, usecols=parameter_columns)
    chosen_df: pd.DataFrame = pd.read_csv(chosen_model_file, usecols=chosen_columns)

    validate_unique_families(parameter_df, fitted_parameters_file)
    validate_unique_families(chosen_df, chosen_model_file)

    plotting_df: pd.DataFrame = parameter_df.merge(chosen_df, on=FAMILY_NAME_COL, how="outer", validate="one_to_one")

    numeric_columns: list[str] = [
        TREE_SCALING_FACTOR_COL,
        CONSTANT_RATE_COL,
        LINEAR_P1_COL,
        LINEAR_P2_COL,
        EXP_P1_COL,
        EXP_P2_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
        CONSTANT_AICC_COL,
        LINEAR_AICC_COL,
        EXP_AICC_COL,
        IGNORE_AICC_COL,
    ]

    for column in numeric_columns:
        plotting_df[column] = pd.to_numeric(plotting_df[column], errors="coerce")

    required_columns: list[str] = [
        CHOSEN_FUNCTION_LABEL_COL,
        CONSTANT_RATE_COL,
        LINEAR_P1_COL,
        LINEAR_P2_COL,
        EXP_P1_COL,
        EXP_P2_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
    ]
    missing_mask: pd.Series = plotting_df[required_columns].isna().any(axis=1)

    if missing_mask.any():
        missing_families: list[str] = plotting_df.loc[missing_mask, FAMILY_NAME_COL].astype(str).tolist()
        raise ValueError(f"Families missing model-selection or fitted-function information: {missing_families}")

    valid_models: set[str] = {LABEL_CONSTANT, LABEL_LINEAR, LABEL_EXP, LABEL_IGNORE}
    invalid_model_mask: pd.Series = ~plotting_df[CHOSEN_FUNCTION_LABEL_COL].isin(valid_models)

    if invalid_model_mask.any():
        invalid_values: list[str] = plotting_df.loc[invalid_model_mask, CHOSEN_FUNCTION_LABEL_COL].astype(str).unique().tolist()
        raise ValueError(f"Unexpected chosen-model labels: {invalid_values}")

    return plotting_df.sort_values(FAMILY_NAME_COL).reset_index(drop=True)


def format_aicc_ranking_text(row: pd.Series) -> str:
    ranking: list[tuple[str, float]] = []

    for model_label in MODEL_LABELS_WITH_EXP:
        aicc_value: float = to_float(row.get(AICC_COL_BY_MODEL[model_label], np.nan))

        if not pd.isna(aicc_value):
            ranking.append((model_label, aicc_value))

    ranking.sort(key=lambda item: item[1])

    if not ranking:
        return "AICc ranking\nmissing"

    lines: list[str] = ["AICc ranking"]
    lines.extend(f"{rank}. {model_label:<11} {aicc_value:.3f}" for rank, (model_label, aicc_value) in enumerate(ranking, start=1))
    return "\n".join(lines)


def plot_family_functions(row: pd.Series, transition_label: str, output_dir: Path) -> Path:
    family_name: str = str(row[FAMILY_NAME_COL])
    chosen_model: str = str(row[CHOSEN_FUNCTION_LABEL_COL])
    min_chr: float = to_float(row[MIN_CHROM_COL])
    max_chr: float = to_float(row[MAX_CHROM_COL])
    chromosome_grid: np.ndarray | None = build_chromosome_grid(min_chr, max_chr)

    if chromosome_grid is None:
        raise ValueError(f"Could not construct chromosome grid for {family_name}: min={min_chr}, max={max_chr}")

    curves: list[tuple[str, np.ndarray]] = [
        (
            LABEL_CONSTANT,
            np.asarray(compute_constant_rate(chromosome_grid, to_float(row[CONSTANT_RATE_COL])), dtype=float),
        ),
        (
            LABEL_LINEAR,
            np.asarray(compute_linear_rate(chromosome_grid, to_float(row[LINEAR_P1_COL]), to_float(row[LINEAR_P2_COL])), dtype=float),
        ),
        (
            LABEL_EXP,
            np.asarray(compute_exp_rate(chromosome_grid, to_float(row[EXP_P1_COL]), to_float(row[EXP_P2_COL])), dtype=float),
        ),
    ]

    non_finite_models: list[str] = [model_label for model_label, rates in curves if not np.isfinite(rates).all()]
    if non_finite_models:
        raise ValueError(f"Non-finite fitted rates for {family_name}: {non_finite_models}")

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_label, rates in curves:
        is_chosen: bool = model_label == chosen_model
        legend_label: str = f"{model_label.upper()} (chosen)" if is_chosen else model_label.upper()

        ax.plot(
            chromosome_grid,
            rates,
            label=legend_label,
            color=MODEL_COLOR_MAP[model_label],
            linestyle=CHOSEN_LINESTYLE if is_chosen else OTHER_LINESTYLE,
            linewidth=CHOSEN_LINEWIDTH if is_chosen else OTHER_LINEWIDTH,
            alpha=1.0 if is_chosen else 0.8,
        )

    ax.set_xlim(0, max_chr)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{transition_label}: {family_name}\nChosen model: {chosen_model.upper()}")
    ax.set_xlabel("Chromosome number")
    ax.set_ylabel("Inferred transition rate")
    ax.legend(loc="upper left")

    ax.text(
        0.98,
        0.98,
        format_aicc_ranking_text(row),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.92),
    )

    output_file: Path = output_dir / f"{transition_label}_{family_name}_function_curves.png"
    save_close(fig, output_file)
    return output_file


def run_per_family_function_plots(transition_label: str, chosen_model_file: Path, fitted_parameters_file: Path, output_dir: Path) -> list[Path]:
    ensure_dir(output_dir)
    plotting_df: pd.DataFrame = load_plotting_table(chosen_model_file, fitted_parameters_file)
    return [plot_family_functions(row, transition_label, output_dir) for _, row in plotting_df.iterrows()]


def main() -> None:
    args: argparse.Namespace = parse_args()
    output_files: list[Path] = run_per_family_function_plots(args.transition_label, args.chosen_model_file, args.fitted_parameters_file, args.output_dir)

    log_run(
        step="per_family_all_fitted_functions",
        script=Path(__file__),
        params={
            "transition_label": args.transition_label,
            "chosen_model_file": args.chosen_model_file,
            "fitted_parameters_file": args.fitted_parameters_file,
            "output_dir": args.output_dir,
            "number_of_generated_plots": len(output_files),
        },
        outputs=[output_file.as_posix() for output_file in output_files],
        description="Plotted tree-scale-corrected CONSTANT, LINEAR, and EXPONENTIAL functions separately for every family.",
        notes="The integrated chosen function was solid and all other fitted functions were dashed. IGNORE-chosen families therefore have three dashed curves.",
    )

    print("[✓] Per-family fitted-function plots completed.")
    print(f"Plots generated: {len(output_files)}")
    print(args.output_dir)


if __name__ == "__main__":
    main()