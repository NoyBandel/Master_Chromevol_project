"""
Add tree-scale-corrected parameters to configured analysis CSV files.

Expected JSON format:

{
  "corrected_suffix": "_tree_scale_corrected",
  "targets": [
    {
      "file": "analysis/baseline_models/gain/gain_chosen_model_table.csv",
      "columns_to_correct": [
        "constant_value",
        "lin_p1",
        "lin_slope_p2"
      ]
    },
    {
      "file": "analysis/exponential_vs_linear/gain/slope_analysis/tables/gain_exp_lin_slope_plotting_table.csv",
      "columns_to_correct": [
        "exp_p1",
        "exp_effective_slope",
        "lin_p1",
        "lin_slope_p2"
      ]
    }
  ]
}

Paths in the JSON may be absolute or relative to PROJECT_ROOT.
The tree-scaling CSV is provided using --scaling-values-file and must contain:
    family_name
    tree_scaling_factor
Additional columns, such as original_tree_length and scaled_tree_length, are ignored.

For every column listed under "columns_to_correct": corrected value = raw value * tree_scaling_factor

Rate-scale parameters that should normally be listed:
    - constant_value
    - lin_p1
    - lin_slope_p2
    - exp_p1
    - linear_effective_slope
    - exp_effective_slope

Exponential shape parameters should not be listed:
    - exp_p2
    - exp_slope_p2

The script preserves the raw columns, adds tree_scaling_factor after family_name, adds each corrected column after its raw column,
and replaces previously generated corrected columns when rerun.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from source_code.constants import ANALYSIS_DIR, FAMILY_NAME_COL, PROJECT_ROOT
from source_code.logger import log_run


TREE_SCALING_FACTOR_COL: str = "tree_scaling_factor"
DEFAULT_CONFIG_FILE: Path = ANALYSIS_DIR / "tree_scaling" / "tree_scale_correction_config.json"
CORRECTION_SUMMARY_FILE: Path = ANALYSIS_DIR / "tree_scaling" / "tree_scale_correction_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add tree-scale-corrected parameters to configured analysis CSV files.")
    parser.add_argument("--scaling-values-file", type=Path, required=True, help="CSV containing family_name and tree_scaling_factor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="Path to the correction JSON configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare corrected tables without overwriting files.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_file: Path) -> dict[str, Any]:
    with config_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def corrected_column_name(column: str, suffix: str) -> str:
    return f"{column}{suffix}"


def read_scaling_table(file_path: Path) -> pd.DataFrame:
    scaling_df: pd.DataFrame = pd.read_csv(file_path, usecols=[FAMILY_NAME_COL, TREE_SCALING_FACTOR_COL])
    scaling_df[TREE_SCALING_FACTOR_COL] = pd.to_numeric(scaling_df[TREE_SCALING_FACTOR_COL], errors="raise")
    return scaling_df


def prepare_corrected_table(target: dict[str, Any], scaling_df: pd.DataFrame, suffix: str) -> tuple[Path, pd.DataFrame]:
    file_path: Path = resolve_path(target["file"])
    columns_to_correct: list[str] = target["columns_to_correct"]
    corrected_columns: list[str] = [corrected_column_name(column, suffix) for column in columns_to_correct]

    target_df: pd.DataFrame = pd.read_csv(file_path)
    target_df = target_df.drop(columns=[TREE_SCALING_FACTOR_COL, *corrected_columns], errors="ignore")
    target_df = target_df.merge(scaling_df, on=FAMILY_NAME_COL, how="left", validate="many_to_one", sort=False)

    missing_families: list[str] = sorted(target_df.loc[target_df[TREE_SCALING_FACTOR_COL].isna(), FAMILY_NAME_COL].drop_duplicates().astype(str).tolist())

    if missing_families:
        raise ValueError(f"Missing tree-scaling factors for families in {file_path}: {missing_families}")

    for column, corrected_column in zip(columns_to_correct, corrected_columns):
        target_df[corrected_column] = pd.to_numeric(target_df[column], errors="raise") * target_df[TREE_SCALING_FACTOR_COL]

    generated_columns: set[str] = {TREE_SCALING_FACTOR_COL, *corrected_columns}
    ordered_columns: list[str] = []

    for column in target_df.columns:
        if column in generated_columns:
            continue

        ordered_columns.append(column)

        if column == FAMILY_NAME_COL:
            ordered_columns.append(TREE_SCALING_FACTOR_COL)

        if column in columns_to_correct:
            ordered_columns.append(corrected_column_name(column, suffix))

    return file_path, target_df[ordered_columns]


def prepare_all_tables(config: dict[str, Any], scaling_df: pd.DataFrame, status: str) -> tuple[dict[Path, pd.DataFrame], pd.DataFrame]:
    prepared_tables: dict[Path, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    suffix: str = config["corrected_suffix"]

    for target in config["targets"]:
        file_path, corrected_df = prepare_corrected_table(target, scaling_df, suffix)
        prepared_tables[file_path] = corrected_df
        summary_rows.append({
            "file_path": file_path.as_posix(),
            "number_of_rows": len(corrected_df),
            "number_of_families": corrected_df[FAMILY_NAME_COL].nunique(),
            "corrected_columns": ";".join(corrected_column_name(column, suffix) for column in target["columns_to_correct"]),
            "status": status,
        })

    return prepared_tables, pd.DataFrame(summary_rows)


def write_outputs(output_tables: dict[Path, pd.DataFrame]) -> None:
    temporary_files: dict[Path, Path] = {}

    try:
        for output_path, output_df in output_tables.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path = output_path.with_name(f".{output_path.name}.tree_scale_tmp")
            temporary_path.unlink(missing_ok=True)
            output_df.to_csv(temporary_path, index=False)
            temporary_files[output_path] = temporary_path

        for output_path, temporary_path in temporary_files.items():
            temporary_path.replace(output_path)

    except Exception:
        for temporary_path in temporary_files.values():
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args: argparse.Namespace = parse_args()
    config_file: Path = resolve_path(args.config)
    scaling_values_file: Path = resolve_path(args.scaling_values_file)
    config: dict[str, Any] = load_config(config_file)
    scaling_df: pd.DataFrame = read_scaling_table(scaling_values_file)

    prepared_tables, summary_df = prepare_all_tables(config, scaling_df, "validated" if args.dry_run else "updated")

    if args.dry_run:
        print(f"Dry run completed successfully for {len(prepared_tables)} target files.")
        print("No files were overwritten.")
        print(summary_df.to_string(index=False))
        return

    output_tables: dict[Path, pd.DataFrame] = {**prepared_tables, CORRECTION_SUMMARY_FILE: summary_df}
    write_outputs(output_tables)

    log_run(
        step="tree_scale_correction",
        script=Path(__file__),
        params={
            "config_file": config_file,
            "scaling_values_file": scaling_values_file,
            "number_of_target_files": len(config["targets"]),
            "corrected_suffix": config["corrected_suffix"],
        },
        outputs=[path.as_posix() for path in output_tables],
        description="Added tree-scaling factors and tree-scale-corrected parameters to the configured analysis tables.",
        notes="Each configured parameter was multiplied by tree_scaling_factor. Raw parameter columns were preserved.",
    )

    print(f"Updated {len(prepared_tables)} analysis CSV files.")
    print(f"Correction summary saved to: {CORRECTION_SUMMARY_FILE}")


if __name__ == "__main__":
    main()