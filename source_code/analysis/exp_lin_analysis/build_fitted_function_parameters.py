"""
Build a complete fitted-function parameter table for the EXP-LINEAR analysis.

The output contains the tree-scale-corrected parameters required to plot the
CONSTANT, LINEAR, and EXPONENTIAL functions for every family.

Parameter sources:
    Baseline chosen-model table:
        constant_value_tree_scale_corrected

    EXP-LINEAR core table:
        lin_p1_tree_scale_corrected
        lin_slope_p2_tree_scale_corrected
        exp_p1_tree_scale_corrected
        exp_slope_p2

    EXP-LINEAR slope plotting table:
        min_chrom
        max_chrom
        model AICc values
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from source_code.analysis.exp_lin_analysis.exp_lin_analysis_constants import CONSTANT_AICC_COL, EXP_AICC_COL, EXP_P2_COL, IGNORE_AICC_COL, LINEAR_AICC_COL
from source_code.analysis.plot_utils import ensure_dir
from source_code.constants import FAMILY_NAME_COL, MAX_CHROM_COL, MIN_CHROM_COL
from source_code.logger import log_run


TREE_SCALING_FACTOR_COL: str = "tree_scaling_factor"
BASELINE_TREE_SCALING_FACTOR_COL: str = "baseline_tree_scaling_factor"
CORE_TREE_SCALING_FACTOR_COL: str = "core_tree_scaling_factor"

CONSTANT_RATE_COL: str = "constant_value_tree_scale_corrected"
LINEAR_P1_COL: str = "lin_p1_tree_scale_corrected"
LINEAR_P2_COL: str = "lin_slope_p2_tree_scale_corrected"
EXP_P1_COL: str = "exp_p1_tree_scale_corrected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the complete fitted-function parameter table.")
    parser.add_argument("--baseline-chosen-model-file", type=Path, required=True, help="Baseline chosen-model table containing the corrected constant rate.")
    parser.add_argument("--core-table-file", type=Path, required=True, help="EXP-LINEAR core table containing corrected LINEAR and EXP parameters.")
    parser.add_argument("--slope-plotting-table-file", type=Path, required=True, help="EXP-LINEAR slope plotting table containing chromosome ranges and AICc values.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output fitted-function parameter CSV.")
    return parser.parse_args()


def read_family_table(file_path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing input file: {file_path}")

    df: pd.DataFrame = pd.read_csv(file_path)
    missing_columns: list[str] = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in {file_path}: {missing_columns}")

    df = df[required_columns].copy()
    df[FAMILY_NAME_COL] = df[FAMILY_NAME_COL].astype("string").str.strip()

    invalid_family_mask: pd.Series = df[FAMILY_NAME_COL].isna() | df[FAMILY_NAME_COL].eq("")
    if invalid_family_mask.any():
        raise ValueError(f"Missing family names in {file_path}: {int(invalid_family_mask.sum())} rows")

    duplicate_mask: pd.Series = df[FAMILY_NAME_COL].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_families: list[str] = sorted(df.loc[duplicate_mask, FAMILY_NAME_COL].astype(str).unique().tolist())
        raise ValueError(f"Duplicate families in {file_path}: {duplicate_families}")

    return df


def validate_complete_values(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_mask: pd.Series = df[required_columns].isna().any(axis=1)

    if not missing_mask.any():
        return

    missing_details: list[str] = []

    for _, row in df.loc[missing_mask, [FAMILY_NAME_COL, *required_columns]].iterrows():
        missing_columns: list[str] = [column for column in required_columns if pd.isna(row[column])]
        missing_details.append(f"{row[FAMILY_NAME_COL]}: {missing_columns}")

    raise ValueError("Missing required fitted-function information:\n" + "\n".join(missing_details))


def build_fitted_function_parameters(baseline_chosen_model_file: Path, core_table_file: Path, slope_plotting_table_file: Path) -> pd.DataFrame:
    baseline_df: pd.DataFrame = read_family_table(
        baseline_chosen_model_file,
        [FAMILY_NAME_COL, TREE_SCALING_FACTOR_COL, CONSTANT_RATE_COL],
    )

    core_df: pd.DataFrame = read_family_table(
        core_table_file,
        [FAMILY_NAME_COL, TREE_SCALING_FACTOR_COL, LINEAR_P1_COL, LINEAR_P2_COL, EXP_P1_COL, EXP_P2_COL],
    )

    slope_df: pd.DataFrame = read_family_table(
        slope_plotting_table_file,
        [
            FAMILY_NAME_COL,
            MIN_CHROM_COL,
            MAX_CHROM_COL,
            CONSTANT_AICC_COL,
            LINEAR_AICC_COL,
            EXP_AICC_COL,
            IGNORE_AICC_COL,
        ],
    )

    baseline_df = baseline_df.rename(columns={TREE_SCALING_FACTOR_COL: BASELINE_TREE_SCALING_FACTOR_COL})
    core_df = core_df.rename(columns={TREE_SCALING_FACTOR_COL: CORE_TREE_SCALING_FACTOR_COL})

    parameter_df: pd.DataFrame = baseline_df.merge(core_df, on=FAMILY_NAME_COL, how="outer", validate="one_to_one")
    parameter_df = parameter_df.merge(slope_df, on=FAMILY_NAME_COL, how="outer", validate="one_to_one")

    numeric_columns: list[str] = [
        BASELINE_TREE_SCALING_FACTOR_COL,
        CORE_TREE_SCALING_FACTOR_COL,
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
        parameter_df[column] = pd.to_numeric(parameter_df[column], errors="coerce")

    required_parameter_columns: list[str] = [
        BASELINE_TREE_SCALING_FACTOR_COL,
        CORE_TREE_SCALING_FACTOR_COL,
        CONSTANT_RATE_COL,
        LINEAR_P1_COL,
        LINEAR_P2_COL,
        EXP_P1_COL,
        EXP_P2_COL,
        MIN_CHROM_COL,
        MAX_CHROM_COL,
    ]
    validate_complete_values(parameter_df, required_parameter_columns)

    scaling_match: np.ndarray = np.isclose(
        parameter_df[BASELINE_TREE_SCALING_FACTOR_COL].to_numpy(dtype=float),
        parameter_df[CORE_TREE_SCALING_FACTOR_COL].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )

    if not scaling_match.all():
        mismatched_families: list[str] = parameter_df.loc[~scaling_match, FAMILY_NAME_COL].astype(str).tolist()
        raise ValueError(f"Tree-scaling factors differ between the baseline and EXP-LINEAR tables: {mismatched_families}")

    invalid_range_mask: pd.Series = (parameter_df[MIN_CHROM_COL] < 1) | (parameter_df[MAX_CHROM_COL] < parameter_df[MIN_CHROM_COL])
    if invalid_range_mask.any():
        invalid_families: list[str] = parameter_df.loc[invalid_range_mask, FAMILY_NAME_COL].astype(str).tolist()
        raise ValueError(f"Invalid chromosome ranges: {invalid_families}")

    parameter_df[TREE_SCALING_FACTOR_COL] = parameter_df[CORE_TREE_SCALING_FACTOR_COL]

    output_columns: list[str] = [
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

    return parameter_df[output_columns].sort_values(FAMILY_NAME_COL).reset_index(drop=True)


def main() -> None:
    args: argparse.Namespace = parse_args()
    parameter_df: pd.DataFrame = build_fitted_function_parameters(args.baseline_chosen_model_file, args.core_table_file, args.slope_plotting_table_file)

    ensure_dir(args.output_file.parent)
    parameter_df.to_csv(args.output_file, index=False)

    log_run(
        step="build_fitted_function_parameters",
        script=Path(__file__),
        params={
            "baseline_chosen_model_file": args.baseline_chosen_model_file,
            "core_table_file": args.core_table_file,
            "slope_plotting_table_file": args.slope_plotting_table_file,
            "output_file": args.output_file,
            "number_of_families": len(parameter_df),
        },
        outputs=[args.output_file.as_posix()],
        description="Built a complete table of tree-scale-corrected CONSTANT, LINEAR, and EXPONENTIAL fitted-function parameters.",
        notes="Tree-scaling factors were validated between the baseline and EXP-LINEAR source tables.",
    )

    print("[✓] Fitted-function parameter table created.")
    print(f"Families: {len(parameter_df)}")
    print(args.output_file)


if __name__ == "__main__":
    main()